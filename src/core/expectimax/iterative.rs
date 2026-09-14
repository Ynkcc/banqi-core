//! 迭代加深：根层单层搜索与逐层加深，维护用于排序的 hint。

use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::env::DarkChessEnv;

use super::config::SearchConfig;
use super::negamax::{move_to_front, move_value};
use super::nnue::NnueAccumulator;
use super::ordering;
use super::zobrist::{INF, VMAX, VMIN};
use super::Ctx;

/// 单层根搜索：返回 (最优动作, 根走子方视角值)。
fn best_at_depth(
    env: &DarkChessEnv,
    acc: &dyn NnueAccumulator,
    depth: i32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
    hint: Option<usize>,
) -> Result<Option<(usize, f32)>, ()> {
    let mut moves = env.generate_moves(env.get_current_player());
    if moves.is_empty() {
        return Ok(None);
    }
    ordering::order_moves(env, &mut moves, depth, cfg, ctx);
    if let Some(h) = hint {
        move_to_front(&mut moves, h);
    }
    let mut best_val = -INF;
    let mut best = None;
    let mut alpha = VMIN;
    for &m in &moves {
        let v = move_value(env, acc, m.action, depth, alpha, VMAX, cfg, ctx)?;
        if v > best_val {
            best_val = v;
            best = Some(m.action);
            if v > alpha {
                alpha = v;
            }
        }
    }
    Ok(best.map(|a| (a, best_val)))
}

/// 迭代加深主循环；`stop` 为 Some 时每层开始前检查停止标志。
///
/// 返回 `(hint, best_action, best_score, depth_reached)`：
/// 助线程只需关心 hint，主搜索消费其余字段。
pub(super) fn iterative_deepen(
    env: &DarkChessEnv,
    acc: &dyn NnueAccumulator,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
    mut hint: Option<usize>,
    stop: Option<&AtomicBool>,
) -> (Option<usize>, Option<usize>, f32, i32) {
    let mut best_action = None;
    let mut best_score = 0.0f32;
    let mut depth_reached = 0;
    for depth in 1..=cfg.max_depth {
        if stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
            break;
        }
        match best_at_depth(env, acc, depth, cfg, &mut *ctx, hint) {
            Ok(Some((a, v))) => {
                hint = Some(a);
                best_action = Some(a);
                best_score = v;
                depth_reached = depth;
            }
            _ => break,
        }
    }
    (hint, best_action, best_score, depth_reached)
}
