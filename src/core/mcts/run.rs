// src/mcts/run.rs
// Gumbel MCTS 单树搜索主循环：Sequential Halving 预算排程与候选淘汰。
//
// 分层说明：
// - 本文件只负责「主循环编排」，具体下探/回填/根准备分别见
//   path_select.rs / root.rs / tree.rs。

use crate::core::env::GameEnv;

use super::budget::SequentialHalvingBudget;
use super::config::MctsSearchResult;
use super::evaluator::{Evaluator, EvaluatorError};
use super::path::{PendingEval, SelectPathOutcome};
use super::search::GumbelMCTS;

impl<'a, G: GameEnv, E: Evaluator<G>> GumbelMCTS<'a, G, E> {
    /// 执行 Gumbel MCTS 搜索主循环
    ///
    /// 1. 扩展根节点。
    /// 2. 收集根节点 Logits 并进行 Gumbel Top-K 采样，选出候选动作。
    /// 3. 使用 Sequential Halving 算法，分阶段分配搜索预算，淘汰表现不佳的候选动作。
    /// 4. 最终返回搜索结果，包含选择的动作和所有相关数据。
    ///
    /// # 返回
    ///
    /// * `Ok(Some(..))` - 正常搜索结果；
    /// * `Ok(None)` - 无合法动作（终局）；
    /// * `Err(..)` - 评估失败（推理后端报错 / 输出不符合契约），由调用方决定重试或终止。
    pub fn run(&mut self) -> Result<Option<MctsSearchResult>, EvaluatorError> {
        // 1. 扩展根节点
        self.expand_root()?;

        self.refresh_root_action_mask();
        if self.root_action_mask.iter().all(|&x| x == 0) {
            return Ok(None);
        }

        // 2. Gumbel-Top-K 采样根候选动作
        let candidates = self.sample_root_candidates();
        if candidates.is_empty() {
            return Ok(None);
        }
        if candidates.len() == 1 {
            // 只有一个候选动作，直接返回
            return Ok(self.build_result(candidates[0]));
        }

        // 3. Sequential Halving
        let mut budget = SequentialHalvingBudget::new(
            candidates.len(),
            self.config.num_simulations,
            2, // eta = 2，表示每阶段淘汰 50% 的动作
        );

        let mut remaining = candidates;

        for phase in 0..budget.num_phases() {
            if remaining.len() <= 1 {
                break;
            }

            let visits_per_action = budget.visits_per_action_in_phase(phase);
            let (phase_usage, terminal_hits) = self.run_phase(visits_per_action, &remaining)?;
            budget.record_phase_usage(phase_usage);

            if phase_usage == 0 {
                // visits_per_action == 0：预算排程自然耗尽（remaining_budget <
                // num_actions），属正常退出，静默 break 不打印告警。
                if visits_per_action == 0 {
                    break;
                }
                // 有预算却无产出：候选路径全部静默早退。若终局回传占满全部调用，
                // 说明子树已全部命中终局（如接近判和/截断阈值的局面），属退化但
                // 良性，静默继续不打印告警，避免训练循环刷屏。
                //
                // 注意：**不在此处 break**。若直接提前退出，剩余候选未按 completed_Q
                // 淘汰，最终会退回 remaining[0] = Gumbel 噪声采样的随机候选。
                self.report_idle_phase(phase, visits_per_action, remaining.len(), terminal_hits);
            }

            // 根据补全复合效用（completed utility）排序并淘汰
            if remaining.len() > 1 {
                remaining = self.eliminate_candidates(&remaining, &budget);
            }

            budget.advance_phase();
        }

        // 4. 收集所有数据并返回
        let Some(action) = self.final_action(&remaining) else {
            return Ok(None);
        };
        Ok(self.build_result(action))
    }

    /// 执行一个阶段：对每个候选各 select 一次为「一轮」，重复 `visits_per_action` 轮。
    ///
    /// 返回 `(实际模拟数, 终局回传次数)`。预算按模拟次数计量：一次
    /// select_path_collect 调用 = 1 次模拟；机会节点展开爆发的 N 路批量评估
    /// 真实发生但不占用模拟预算（否则 chance 子树密集的候选会窃取后续阶段预算）。
    fn run_phase(
        &mut self,
        visits_per_action: usize,
        remaining: &[usize],
    ) -> Result<(usize, usize), EvaluatorError> {
        let mut phase_usage = 0;
        let mut terminal_hits = 0;

        for _ in 0..visits_per_action {
            let mut batch: Vec<PendingEval<G>> = Vec::new();
            let mut eval_calls = 0;
            for &action in remaining {
                match self.select_path_collect(action, &mut batch) {
                    SelectPathOutcome::TerminalBackprop => terminal_hits += 1,
                    SelectPathOutcome::Normal => eval_calls += 1,
                    SelectPathOutcome::EarlyReturn => {}
                }
            }

            if !batch.is_empty() {
                phase_usage += eval_calls;
                self.evaluate_and_apply(&batch)?;
            }
        }

        Ok((phase_usage, terminal_hits))
    }

    /// 批量评估收集到的叶子并回填（单次 `evaluator.evaluate`）。
    fn evaluate_and_apply(&mut self, batch: &[PendingEval<G>]) -> Result<(), EvaluatorError> {
        let envs: Vec<G> = batch.iter().map(|pending| pending.env).collect();
        let out = self.evaluator.evaluate(&envs)?;
        let evals: Vec<(&PendingEval<G>, &[f32], f32, f32)> = batch
            .iter()
            .enumerate()
            .map(|(idx, pending)| {
                let health_mu = if self.config.health_enabled {
                    out.health_expectation(idx).unwrap_or(0.0)
                } else {
                    0.0
                };
                (pending, &out.logits[idx][..], out.values[idx], health_mu)
            })
            .collect();
        self.apply_leaf_evals(&evals);
        Ok(())
    }

    /// 按补全复合效用降序淘汰，保留 `budget.keep_count_after_phase()` 个候选。
    fn eliminate_candidates(
        &self,
        remaining: &[usize],
        budget: &SequentialHalvingBudget,
    ) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = remaining
            .iter()
            .map(|&a| (a, self.completed_utility(a)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let keep_count = budget.keep_count_after_phase();
        scored.into_iter().take(keep_count).map(|(a, _)| a).collect()
    }

    /// 空转阶段告警：仅当存在意外静默早退（终局回传未占满全部调用）时打印。
    fn report_idle_phase(
        &self,
        phase: usize,
        visits_per_action: usize,
        remaining: usize,
        terminal_hits: usize,
    ) {
        let total_calls = visits_per_action * remaining;
        if terminal_hits < total_calls {
            eprintln!(
                "⚠️ MCTS: phase {} 实际模拟数为 0 (visits_per_action={}, remaining={}, 终局回传 {}/{})，按现有 completed_Q 淘汰继续",
                phase, visits_per_action, remaining, terminal_hits, total_calls
            );
        }
    }

    /// 终局动作选择：候选耗尽时退回访问次数最高的子动作。
    fn final_action(&self, remaining: &[usize]) -> Option<usize> {
        if !remaining.is_empty() {
            return Some(remaining[0]);
        }
        let root = self.arena.get(self.root_idx);
        root.children
            .iter()
            .max_by_key(|(_, child_idx)| self.arena.get(*child_idx).visit_count)
            .map(|(action, _)| *action)
    }
}
