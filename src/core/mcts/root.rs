// src/mcts/root.rs
// 根节点准备、叶子批量回填与结果组装（泛型化：G = 游戏环境）
//
// 分层说明：
// - 本文件集中「根评估 → 候选采样 → 叶子回填 → 结果组装」的根侧逻辑，
//   由 run.rs（单树主循环）与 batched.rs（批量自对弈）共用；
// - 路径选择见 path_select.rs，树构建与价值回溯见 tree.rs。

use crate::core::env::GameEnv;

use super::config::MctsSearchResult;
use super::evaluator::{Evaluator, EvaluatorError};
use super::node::value_from_perspective;
use super::path::PendingEval;
use super::search::GumbelMCTS;

impl<'a, G: GameEnv, E: Evaluator<G>> GumbelMCTS<'a, G, E> {
    /// 计算补全后的 Q 值 (Completed Q-value)
    ///
    /// 用于在 Sequential Halving 过程中评估动作优劣。
    ///
    /// 规则:
    /// - N > 0 时：使用 W / N
    /// - N = 0 时：使用网络预测的 initial_value，或已访问兄弟子节点的平均 Q
    /// - 根节点不存在该子动作时：返回 0.0（中性）
    pub(crate) fn completed_q(&self, action: usize) -> f32 {
        let root = self.arena.get(self.root_idx);
        if let Some(child_idx) = root.child_idx(action) {
            let child_player = self.arena.get(child_idx).player();
            let q = self.node_q_value(child_idx);
            // 统一到根玩家视角：翻子动作的 child 为机会节点（未执行 step，
            // player == root.player），视角天然一致；移动/炮击动作的 child
            // 已执行 step（player 为对手），Q 符号需取反。
            value_from_perspective(root.player, child_player, q)
        } else {
            0.0
        }
    }

    /// 获取根节点指定动作的 completed_Q
    pub fn get_root_completed_q(&self, action: usize) -> f32 {
        self.completed_q(action)
    }

    /// 计算根节点指定动作的补全复合效用（completed utility），用于 Sequential
    /// Halving 淘汰。与 `completed_q`（纯胜率，作为训练目标）解耦：health_enabled
    /// 时并入血量期望，否则与 `completed_q` 一致。
    pub(crate) fn completed_utility(&self, action: usize) -> f32 {
        let root = self.arena.get(self.root_idx);
        if let Some(child_idx) = root.child_idx(action) {
            let child_player = self.arena.get(child_idx).player();
            let u = self.node_utility_value(child_idx);
            value_from_perspective(root.player, child_player, u)
        } else {
            0.0
        }
    }

    /// 展开根节点
    ///
    /// 在搜索开始前，确保根节点已经被评估和扩展。
    /// 根节点缺失环境（不应发生）时打印错误并跳过，交由调用方按空掩码处理。
    pub(crate) fn expand_root(&mut self) -> Result<(), EvaluatorError> {
        if self.arena.get(self.root_idx).is_expanded {
            return Ok(());
        }

        let Some(env) = self.root_env_copied() else {
            eprintln!("⚠️ MCTS: 根节点缺少环境，跳过根展开");
            return Ok(());
        };
        let out = self.evaluator.evaluate(std::slice::from_ref(&env))?;
        let health_mu = if self.config.health_enabled {
            out.health_expectation(0).unwrap_or(0.0)
        } else {
            0.0
        };
        self.apply_root_eval(&out.logits[0], out.values[0], health_mu);
        Ok(())
    }

    /// 刷新根动作掩码（run 与 batched 根准备共用）。
    /// 根节点缺失环境时掩码保持全 0（调用方据此返回 None）。
    pub(crate) fn refresh_root_action_mask(&mut self) {
        self.root_action_mask.iter_mut().for_each(|m| *m = 0);
        let Some(env) = self.root_env_copied() else {
            eprintln!("⚠️ MCTS: 根节点缺少环境，动作掩码置空");
            return;
        };
        env.action_masks_into(&mut self.root_action_mask);
    }

    /// 收集根节点各动作的 logits（未出现的动作填 -1e6）。
    pub(crate) fn root_logits(&self) -> Vec<f32> {
        (0..self.action_space)
            .map(|i| {
                self.arena
                    .get(self.root_idx)
                    .child_idx(i)
                    .map(|idx| self.arena.get(idx).logit)
                    .unwrap_or(-1e6)
            })
            .collect()
    }

    /// Gumbel Top-K 采样根候选动作（run 与 batched 共用；依赖 root_action_mask 已刷新）。
    pub(crate) fn sample_root_candidates(&mut self) -> Vec<usize> {
        let logits = self.root_logits();
        let masks_cloned = self.root_action_mask.clone();
        self.sample_gumbel_top_k(&logits, &masks_cloned, self.config.max_considered_actions)
    }

    /// 根扩展：以评估结果构建子节点并写入初始值 / 访问计数（run 与 batched 共用）。
    pub(crate) fn apply_root_eval(&mut self, logits: &[f32], value: f32, health_mu: f32) {
        let Some(env) = self.root_env_copied() else {
            eprintln!("⚠️ MCTS: 根节点缺少环境，无法应用根评估");
            return;
        };
        let mut masks = vec![0; self.action_space];
        env.action_masks_into(&mut masks);
        let probs = self.compute_probs_from_logits(logits, &masks);

        Self::build_children_from_eval(
            &mut self.arena,
            self.root_idx,
            &env,
            &probs,
            logits,
            value,
            health_mu,
        );

        let root = self.arena.get_mut(self.root_idx);
        root.initial_value = value;
        root.initial_health = health_mu;
        root.visit_count += 1;
        root.value_sum += value;
        root.health_sum += health_mu;
    }

    /// 叶子批量回填：masks→probs→定位叶→写入 initial_value/health→扩展子节点→加权回传。
    ///
    /// `evals` 与最近一次 select_path_collect 收集（或 batched collect）顺序一致。
    /// run 与 batched 共用。
    pub(crate) fn apply_leaf_evals(&mut self, evals: &[(&PendingEval<G>, &[f32], f32, f32)]) {
        let mut eval_values: Vec<(f32, f32)> = Vec::with_capacity(evals.len());
        for (pending, logits, value, health) in evals {
            let mut masks = vec![0; self.action_space];
            pending.env.action_masks_into(&mut masks);
            let probs = self.compute_probs_from_logits(logits, &masks);
            let Some(leaf_idx) = Self::get_node_idx_by_path(&self.arena, self.root_idx, &pending.path)
            else {
                eprintln!("⚠️ MCTS: 叶子路径查找失败，跳过该评估项");
                continue;
            };
            {
                let leaf = self.arena.get_mut(leaf_idx);
                leaf.initial_value = *value;
                leaf.initial_health = *health;
            }
            Self::build_children_from_eval(
                &mut self.arena,
                leaf_idx,
                &pending.env,
                &probs,
                logits,
                *value,
                *health,
            );
            eval_values.push((*value, *health));
        }
        let backprop_evals: Vec<(&PendingEval<G>, f32, f32)> = evals
            .iter()
            .zip(eval_values)
            .map(|((pending, _, _, _), (v, h))| (*pending, v, h))
            .collect();
        Self::backprop_evals(&mut self.arena, self.root_idx, &backprop_evals);
    }

    /// 组装搜索结果（根状态/策略/Q 值/访问计数等统一收集点）。
    /// 根状态缺失（理论上不可能）时返回 None。
    pub(crate) fn build_result(&self, action: usize) -> Option<MctsSearchResult> {
        let root = self.arena.get(self.root_idx);
        let state = root.state.clone()?;
        let player = root.player;
        let improved_policy = self.get_improved_policy();
        let mcts_value = root.q_value();
        let completed_q = self.completed_q(action);
        let root_visit_count = root.visit_count;
        let action_mask = self.root_action_mask.clone();

        Some(MctsSearchResult {
            action,
            state,
            improved_policy,
            mcts_value,
            completed_q,
            root_visit_count,
            player,
            action_mask,
        })
    }
}
