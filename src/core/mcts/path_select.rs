// src/mcts/path_select.rs
// 路径选择：从根候选出发下探至叶子 / 终局 / 机会节点（泛型化：G = 游戏环境）
//
// 分层说明：
// - 本文件只负责「沿树向下选择一条路径并收集待评估项」；
// - 根准备见 root.rs，主循环见 run.rs，树构建与价值回溯见 tree.rs。

use crate::core::env::GameEnv;

use super::evaluator::Evaluator;
use super::node::value_from_perspective;
use super::path::{ChanceSeed, PathStep, PendingEval, SelectPathOutcome};
use super::search::GumbelMCTS;

/// 单次 `select_path_collect` 允许的最大路径步数。
///
/// 正常路径长度受棋盘规模与 `config.max_steps_per_episode` 约束，远小于该值；
/// 该上限仅用于防御极端情况（如树结构损坏导致的无限循环）。
/// 超限时按当前节点已有 Q 值回传兜底，避免静默丢弃路径。
const MAX_SELECT_STEPS: usize = 512;

/// 机会节点处理结果：继续下探到某个 outcome 子节点，或直接返回。
enum ChanceStep {
    Continue(usize),
    Return(SelectPathOutcome),
}

impl<'a, G: GameEnv, E: Evaluator<G>> GumbelMCTS<'a, G, E> {
    /// 选择路径并收集待评估项
    ///
    /// 从根节点的特定动作出发，执行模拟直到到达叶子节点或游戏结束。
    /// 如果到达未扩展的节点，将其加入 `batch` 等待后续评估。
    ///
    /// 模拟过程中使用 PUCT 公式 (Predictor + Upper Confidence Bound applied to Trees) 选择动作：
    /// Score = Q(s, a) + U(s, a)
    /// U(s, a) = c_puct * P(s, a) * sqrt(N(parent)) / (1 + N(child))
    pub(crate) fn select_path_collect(
        &mut self,
        action: usize,
        batch: &mut Vec<PendingEval<G>>,
    ) -> SelectPathOutcome {
        let mut path = vec![PathStep::Action(action)];
        let Some(mut current_idx) = self.arena.get(self.root_idx).child_idx(action) else {
            eprintln!(
                "⚠️ MCTS: select_path_collect 根节点缺少候选动作 {} 的子节点",
                action
            );
            return SelectPathOutcome::EarlyReturn;
        };
        let mut current_action = action;
        let mut steps_taken = 0;

        loop {
            steps_taken += 1;
            if steps_taken > MAX_SELECT_STEPS {
                return self.backprop_steps_limit(current_idx, &path);
            }

            if self.arena.get(current_idx).is_chance_node {
                match self.step_chance_node(current_idx, current_action, &mut path, batch) {
                    ChanceStep::Continue(next_idx) => {
                        current_idx = next_idx;
                        continue;
                    }
                    ChanceStep::Return(outcome) => return outcome,
                }
            }

            let Some(env) = self.arena.get(current_idx).env else {
                eprintln!("⚠️ MCTS: 节点缺少环境 (node={})", current_idx);
                return SelectPathOutcome::EarlyReturn;
            };

            // 使用遍历缓冲 action_mask
            self.traversal_action_mask.iter_mut().for_each(|m| *m = 0);
            env.action_masks_into(&mut self.traversal_action_mask);

            // 终局检测：优先使用节点缓存的 is_terminal（覆盖分数归零/全灭/
            // 无合法动作/连续无吃子判和/步数截断），按真实胜负回传；
            // action_mask 全 0 作为兜底（理论上已包含在 is_terminal 中）。
            if self.arena.get(current_idx).is_terminal
                || self.traversal_action_mask.iter().all(|&x| x == 0)
            {
                return self.backprop_terminal(current_idx, &env, &path);
            }

            if !self.arena.get(current_idx).is_expanded {
                let leaf_player = self.arena.get(current_idx).player();
                batch.push(PendingEval {
                    path,
                    env,
                    leaf_player,
                    chance_seed: None,
                });
                return SelectPathOutcome::Normal;
            }

            let Some((act, next_idx)) = self.puct_select_child(current_idx) else {
                return SelectPathOutcome::EarlyReturn;
            };
            path.push(PathStep::Action(act));
            current_action = act;
            current_idx = next_idx;
        }
    }

    /// 机会节点处理：未展开则全量展开并批量产出 outcome 待评估项；
    /// 已展开则按概率采样一个 outcome 继续下探。
    fn step_chance_node(
        &mut self,
        node_idx: usize,
        action: usize,
        path: &mut Vec<PathStep>,
        batch: &mut Vec<PendingEval<G>>,
    ) -> ChanceStep {
        if !self.arena.get(node_idx).is_expanded {
            Self::expand_chance_node(&mut self.arena, node_idx, action);
            let possible_states = self.arena.get(node_idx).possible_states.clone();

            if possible_states.is_empty() {
                eprintln!(
                    "⚠️ MCTS: chance 节点展开后无可选结果 (node={}, action={})",
                    node_idx, action
                );
                return ChanceStep::Return(SelectPathOutcome::EarlyReturn);
            }

            let base_path = path.clone();
            for (outcome_id, prob, child_idx) in possible_states.iter() {
                let Some(child_env) = self.arena.get(*child_idx).env else {
                    eprintln!("⚠️ MCTS: chance 子节点缺少环境 (node={})", child_idx);
                    return ChanceStep::Return(SelectPathOutcome::EarlyReturn);
                };
                let mut outcome_path = base_path.clone();
                outcome_path.push(PathStep::ChanceOutcome(*outcome_id));
                let leaf_player = self.arena.get(*child_idx).player();
                batch.push(PendingEval {
                    path: outcome_path,
                    env: child_env,
                    leaf_player,
                    chance_seed: Some(ChanceSeed {
                        chance_idx: node_idx,
                        prob: *prob,
                        prefix_len: base_path.len(),
                    }),
                });
            }
            return ChanceStep::Return(SelectPathOutcome::Normal);
        }

        let possible_states = self.arena.get(node_idx).possible_states.clone();
        let Some((outcome_id, child_idx)) =
            Self::sample_outcome(&possible_states, &mut self.rng)
        else {
            eprintln!(
                "⚠️ MCTS: 已展开 chance 节点无结果可采样 (node={}, action={})",
                node_idx, action
            );
            return ChanceStep::Return(SelectPathOutcome::EarlyReturn);
        };
        path.push(PathStep::ChanceOutcome(outcome_id));
        ChanceStep::Continue(child_idx)
    }

    /// 步数上限兜底：不再深入，按当前节点已有 Q 值回传。
    fn backprop_steps_limit(
        &mut self,
        current_idx: usize,
        path: &[PathStep],
    ) -> SelectPathOutcome {
        let leaf_player = self.arena.get(current_idx).player();
        let leaf_value = self.node_q_value(current_idx);
        let leaf_health = self.node_health_value(current_idx);
        let path_clone = path.to_vec();
        Self::backprop_from_path(
            &mut self.arena,
            self.root_idx,
            &path_clone,
            leaf_player,
            leaf_value,
            leaf_health,
        );
        SelectPathOutcome::EarlyReturn
    }

    /// 终局节点回传：按真实胜负与血量差回填。
    fn backprop_terminal(
        &mut self,
        current_idx: usize,
        env: &G,
        path: &[PathStep],
    ) -> SelectPathOutcome {
        let leaf_player = self.arena.get(current_idx).player();
        let (_, _, winner) = env.check_game_over_conditions();
        let leaf_value = match winner {
            Some(w) if w == leaf_player.val() => 1.0,
            Some(w) if w == leaf_player.opposite().val() => -1.0,
            _ => 0.0, // 平局 (Some(0)) 或 winner=None
        };
        // 终局血量期望：整型血量差（红方视角）转到 leaf_player 视角后按 D 归一化。
        let leaf_health = if self.config.health_enabled {
            match (env.terminal_health_diff_red_int(), env.health_diff_scale()) {
                (Some(d), s) if s > 0.0 => {
                    let v = if leaf_player.val() == 1 { d as f32 } else { -(d as f32) };
                    (v / s).clamp(-1.0, 1.0)
                }
                _ => 0.0,
            }
        } else {
            0.0
        };
        let path_clone = path.to_vec();
        Self::backprop_from_path(
            &mut self.arena,
            self.root_idx,
            &path_clone,
            leaf_player,
            leaf_value,
            leaf_health,
        );
        SelectPathOutcome::TerminalBackprop
    }

    /// PUCT 子节点选择：返回 `(动作, 子节点索引)`，无可用子节点时返回 None。
    fn puct_select_child(&self, node_idx: usize) -> Option<(usize, usize)> {
        let current = self.arena.get(node_idx);
        let sqrt_total = (current.visit_count as f32).sqrt();
        let parent_player = current.player();
        let children_clone = current.children.clone();
        let puct_coeff = self.config.c_scale.max(0.1);

        let mut best: Option<(usize, usize)> = None;
        let mut best_score = f32::NEG_INFINITY;
        for (act, child_idx) in children_clone.iter() {
            let child = self.arena.get(*child_idx);
            // 复合效用：health_enabled 时并入血量期望，否则退化为纯胜率 Q。
            let child_utility = self.node_utility_value(*child_idx);
            let child_player = child.player();
            let adjusted_q = value_from_perspective(parent_player, child_player, child_utility);
            let u_score = puct_coeff * child.prior * sqrt_total / (1.0 + child.visit_count as f32);
            let score = adjusted_q + u_score;
            if score > best_score {
                best_score = score;
                best = Some((*act, *child_idx));
            }
        }

        if best.is_none() {
            eprintln!("⚠️ MCTS: 已展开节点无可选子节点 (node={})", node_idx);
        }
        best
    }
}
