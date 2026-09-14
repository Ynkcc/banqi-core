//! Expecti-Alpha-Beta 递归主体：negamax、静态搜索、Star1 机会节点。

use crate::core::env::symmetry::{search_group, Symmetry};
use crate::core::env::{DarkChessEnv, Move};

use super::config::SearchConfig;
use super::eval::{eval_acc, outcome_acc, step_with_acc};
use super::nnue::NnueAccumulator;
use super::ordering;
use super::zobrist::{self, TtEntry, INF, VMAX, VMIN};
use super::{Ctx, FEAT_LMR, FEAT_ORDERING, FEAT_REP, FEAT_TT};

/// 置换表探测结果：可立即截断的值，以及用于排序的 best move 提示。
struct TtProbe {
    cutoff: Option<f32>,
    best_move: Option<usize>,
}

/// 静态搜索：仅延展吃明子走法。
fn quiesce(
    env: &DarkChessEnv,
    acc: &dyn NnueAccumulator,
    mut alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
    qdepth: i32,
) -> Result<f32, ()> {
    ctx.tick()?;
    let moves = env.generate_moves(env.get_current_player());
    if let Some(winner) = ordering::terminal_info(env, &moves) {
        return Ok(ordering::terminal_value(env, Some(winner), cfg, ctx));
    }
    let stand = eval_acc(env, acc, cfg);
    if stand >= beta || qdepth <= 0 {
        return Ok(stand);
    }
    if stand > alpha {
        alpha = stand;
    }
    let mut caps: Vec<(i32, Move)> = moves
        .iter()
        .filter(|m| m.is_capture)
        .map(|&m| (ordering::victim_value(env, &m), m))
        .collect();
    caps.sort_by(|a, b| b.0.cmp(&a.0));
    let mut best = stand;
    for (_, m) in caps {
        let mut child = *env;
        let child_acc = step_with_acc(env, acc, m.action, &mut child);
        let v = -quiesce(&child, &*child_acc, -beta, -alpha, cfg, ctx, qdepth - 1)?;
        if v > best {
            best = v;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    Ok(best)
}

/// Star1 机会节点：按概率加权期望值，用区间边界做剪枝。
fn flip_value(
    env: &DarkChessEnv,
    acc: &dyn NnueAccumulator,
    action: usize,
    depth: i32,
    alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
) -> Result<f32, ()> {
    let outcomes = env.chance_outcomes(action);
    if outcomes.is_empty() {
        return Ok(0.0);
    }
    let (l, u) = (VMIN, VMAX);
    let mut vsum = 0.0f32;
    let mut rem = 1.0f32;
    // 机会节点减深：翻棋结果的信息价值低于决策节点，子分支少搜一层换分支覆盖
    let child_depth = (depth - 1 - cfg.chance_reduction).max(0);
    for (_, p, next_env) in outcomes {
        rem -= p;
        if rem < 0.0 {
            rem = 0.0;
        }
        let ai = (alpha - vsum - rem * u) / p;
        let bi = (beta - vsum - rem * l) / p;
        if ai >= u {
            return Ok(alpha);
        }
        if bi <= l {
            return Ok(beta);
        }
        let cl = if ai > l { ai } else { l };
        let cu = if bi < u { bi } else { u };
        let next_acc = outcome_acc(env, acc, action, &next_env);
        let v = -negamax(&next_env, &*next_acc, child_depth, -cu, -cl, cfg, ctx)?;
        if v <= ai {
            return Ok(alpha);
        }
        if v >= bi {
            return Ok(beta);
        }
        vsum += p * v;
    }
    Ok(vsum)
}

/// 单条走子的值。
pub(super) fn move_value(
    env: &DarkChessEnv,
    acc: &dyn NnueAccumulator,
    action: usize,
    depth: i32,
    alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
) -> Result<f32, ()> {
    if env.is_chance_action(action) {
        return flip_value(env, acc, action, depth, alpha, beta, cfg, ctx);
    }
    let mut child = *env;
    let child_acc = step_with_acc(env, acc, action, &mut child);
    Ok(-negamax(&child, &*child_acc, depth - 1, -beta, -alpha, cfg, ctx)?)
}

/// 置换表探测：命中且深度足够时返回可立即截断的值，否则给出 best move 提示。
///
/// `tt_sym_probe` 开启时，原始键 miss 后用对称视角键二次探测（仅统计，不参与截断）。
fn probe_tt(
    env: &DarkChessEnv,
    key: u64,
    depth: i32,
    alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &Ctx,
) -> TtProbe {
    let mut probe = TtProbe {
        cutoff: None,
        best_move: None,
    };
    if !cfg.feat(FEAT_TT) {
        return probe;
    }
    match ctx.tt.probe(key) {
        Some((value, e_depth, flag, best_hint)) => {
            ctx.tt.bump(1);
            if e_depth >= depth {
                match flag {
                    1 => probe.cutoff = Some(value), // exact
                    2 => {
                        if value >= beta {
                            probe.cutoff = Some(value);
                        }
                    }
                    3 => {
                        if value <= alpha {
                            probe.cutoff = Some(value);
                        }
                    }
                    _ => {}
                }
            }
            probe.best_move = Some(best_hint as usize);
        }
        None => {
            if cfg.tt_sym_probe {
                ctx.tt.bump(0);
                for &sym in search_group(env.config.rows, env.config.cols) {
                    if sym == Symmetry::Identity {
                        continue;
                    }
                    if let Some((_, e_depth, _, _)) = ctx.tt.probe(zobrist::sym_zkey(env, sym)) {
                        ctx.tt.bump(2);
                        if e_depth >= depth {
                            ctx.tt.bump(3);
                        }
                        break;
                    }
                }
            }
        }
    }
    probe
}

/// 按排序后的走子列表搜索，返回 `(最优值, 最优动作)`；命中截断时记录杀手/历史。
fn search_ordered_moves(
    env: &DarkChessEnv,
    acc: &dyn NnueAccumulator,
    ordered: &[Move],
    depth: i32,
    alpha: &mut f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
) -> Result<(f32, usize), ()> {
    let mut best = -INF;
    let mut best_m = ordered.first().map(|m| m.action).unwrap_or(0);
    for (i, &m) in ordered.iter().enumerate() {
        let quiet = !m.is_chance && !m.is_capture;
        let v = if cfg.feat(FEAT_LMR) && quiet && i >= 3 && depth >= 3 {
            let mut child = *env;
            let child_acc = step_with_acc(env, acc, m.action, &mut child);
            let probe = -negamax(
                &child,
                &*child_acc,
                depth - 2,
                -*alpha - 1e-6,
                -*alpha,
                cfg,
                ctx,
            )?;
            if probe > *alpha {
                -negamax(&child, &*child_acc, depth - 1, -beta, -*alpha, cfg, ctx)?
            } else {
                probe
            }
        } else {
            move_value(env, acc, m.action, depth, *alpha, beta, cfg, ctx)?
        };
        if v > best {
            best = v;
            best_m = m.action;
        }
        if best > *alpha {
            *alpha = best;
        }
        if *alpha >= beta {
            if cfg.feat(FEAT_ORDERING) && quiet {
                ctx.record_cutoff(&m, depth, env.config.total_positions);
            }
            break;
        }
    }
    Ok((best, best_m))
}

/// 置换表存储（失败/精确/上界三分；last-write-wins，跨线程安全）。
fn store_tt(
    key: u64,
    best: f32,
    best_m: usize,
    alpha_orig: f32,
    beta: f32,
    depth: i32,
    ctx: &Ctx,
) {
    let flag = if best <= alpha_orig {
        3 // fail-low → 上界
    } else if best >= beta {
        2 // fail-high → 下界
    } else {
        1 // exact
    };
    ctx.tt.store_cond(
        key,
        &TtEntry {
            key,
            value: best,
            depth: depth as i16,
            flag,
            best: best_m,
        },
    );
}

/// 将指定动作的走子移到列表头部（hint / TT best move 置顶）。
pub(super) fn move_to_front(moves: &mut [Move], action: usize) {
    if let Some(pos) = moves.iter().position(|m| m.action == action) {
        let m = moves[pos];
        moves.copy_within(0..pos, 1);
        moves[0] = m;
    }
}

pub(super) fn negamax(
    env: &DarkChessEnv,
    acc: &dyn NnueAccumulator,
    depth: i32,
    mut alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
) -> Result<f32, ()> {
    ctx.tick()?;
    let moves = env.generate_moves(env.get_current_player());
    if let Some(winner) = ordering::terminal_info(env, &moves) {
        return Ok(ordering::terminal_value(env, Some(winner), cfg, ctx));
    }
    if depth <= 0 {
        if cfg.quiesce {
            return quiesce(env, acc, alpha, beta, cfg, ctx, cfg.quiesce_max);
        }
        return Ok(eval_acc(env, acc, cfg));
    }
    let alpha_orig = alpha;
    let key = if cfg.feat(FEAT_TT) || cfg.feat(FEAT_REP) {
        zobrist::zkey(env)
    } else {
        0
    };

    // 重复检测：静走循环会产生相同 zkey（吃子/翻棋改变袋或棋盘 → 键不同）。
    if cfg.feat(FEAT_REP) && ctx.path.contains(&key) {
        return Ok(ordering::terminal_value(env, Some(0), cfg, ctx));
    }

    let probe = probe_tt(env, key, depth, alpha, beta, cfg, ctx);
    if let Some(v) = probe.cutoff {
        return Ok(v);
    }

    let mut ordered = moves;
    ordering::order_moves(env, &mut ordered, depth, cfg, ctx);
    if let Some(tm) = probe.best_move {
        move_to_front(&mut ordered, tm);
    }
    if cfg.feat(FEAT_REP) {
        ctx.path.push(key);
    }

    let (best, best_m) = search_ordered_moves(env, acc, &ordered, depth, &mut alpha, beta, cfg, ctx)?;

    if cfg.feat(FEAT_REP) {
        ctx.path.pop();
    }

    if cfg.feat(FEAT_TT) {
        store_tt(key, best, best_m, alpha_orig, beta, depth, ctx);
    }
    Ok(best)
}
