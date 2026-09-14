//! 叶节点评估与增量累加器辅助。

use crate::core::env::DarkChessEnv;

use super::config::SearchConfig;
use super::nnue::NnueAccumulator;

/// 叶节点局面评估入口（NNUE 为唯一评估来源；搜索入口处强制要求已加载权重）
#[inline]
pub(super) fn eval_state(env: &DarkChessEnv, cfg: &SearchConfig) -> f32 {
    match cfg.nnue_evaluator.as_ref() {
        Some(nnue) => nnue.evaluate(env),
        None => {
            eprintln!("❌ Expectimax 搜索未加载 NNUE 权重（SearchConfig.nnue_evaluator = None），叶评估无效");
            0.0
        }
    }
}

/// 基于双累加器的 O(1) 叶节点评估（当前行棋方视角）。
#[inline]
pub(super) fn eval_acc(env: &DarkChessEnv, acc: &dyn NnueAccumulator, cfg: &SearchConfig) -> f32 {
    match cfg.nnue_evaluator.as_ref() {
        Some(_) => acc.evaluate(env.get_current_player()),
        None => eval_state(env, cfg),
    }
}

/// 在父局面上执行一步动作，生成子局面并增量更新双累加器。
#[inline]
pub(super) fn step_with_acc(
    env: &DarkChessEnv,
    acc: &dyn NnueAccumulator,
    action: usize,
    child: &mut DarkChessEnv,
) -> Box<dyn NnueAccumulator> {
    let before = *env;
    let _ = child.step(action, None);
    let mut child_acc = acc.clone_box();
    child_acc.apply_step(&before, child, action);
    child_acc
}

/// 机会节点结果局面的双累加器（结果环境由 chance_outcomes 产生）。
#[inline]
pub(super) fn outcome_acc(
    env: &DarkChessEnv,
    acc: &dyn NnueAccumulator,
    action: usize,
    next_env: &DarkChessEnv,
) -> Box<dyn NnueAccumulator> {
    let mut child_acc = acc.clone_box();
    child_acc.apply_step(env, next_env, action);
    child_acc
}
