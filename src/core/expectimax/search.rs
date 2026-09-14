//! Expecti-Alpha-Beta 搜索入口与 Lazy SMP 并发。
//!
//! 递归主体（negamax / Star1 / quiescence）见 negamax.rs，
//! 迭代加深见 iterative.rs，叶评估见 eval.rs。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::core::env::DarkChessEnv;

use super::iterative::iterative_deepen;
use super::nnue::NnueEvaluate;
use super::smp::SharedTT;
use super::zobrist;
use super::{Ctx, FEAT_REP, FEAT_TT};

pub use super::config::{SearchConfig, SearchResult};
pub use super::zobrist::{INF, TT_EMPTY, TtEntry, VMAX, VMIN};

/// 校验 NNUE 评估器已注入且特征维度匹配；失败时打印错误并返回 None。
fn require_nnue(
    env: &DarkChessEnv,
    cfg: &SearchConfig,
    missing_msg: &str,
) -> Option<Arc<dyn NnueEvaluate>> {
    let Some(nnue) = cfg.nnue_evaluator.clone() else {
        eprintln!("❌ {missing_msg}");
        return None;
    };
    if !feature_dim_ok(env, &*nnue) {
        return None;
    }
    Some(nnue)
}

/// 特征维度校验，不匹配时打印错误。
fn feature_dim_ok(env: &DarkChessEnv, nnue: &dyn NnueEvaluate) -> bool {
    match nnue.validate_feature_dim(env.config.nnue_feature_dim()) {
        Ok(()) => true,
        Err(msg) => {
            eprintln!("❌ {msg}");
            false
        }
    }
}

/// 节点/时间预算驱动的迭代加深 Expectimax 搜索。返回 `None` 表示无合法动作（终局）。
///
/// 强制要求 `cfg.nnue_evaluator` 已加载：未加载权重时直接拒绝搜索（叶评估
/// 以 NNUE 为唯一来源，不提供规则评估兜底）。
pub fn search(env: &DarkChessEnv, cfg: &SearchConfig) -> Option<SearchResult> {
    require_nnue(
        env,
        cfg,
        "❌ Expectimax 搜索需要 NNUE 评估器：请经 ExpectimaxEngine::from_nnue_file 加载 .nnue 权重（SearchConfig.nnue_evaluator = None）",
    )?;
    let shared = Arc::new(SharedTT::new(cfg.tt_bits));
    search_with_tt(env, cfg, shared)
}

/// Lazy SMP 并发搜索：`threads <= 1` 时等价 `search`；>1 时主线程迭代加深，
/// 助线程以独立上下文共享置换表协同搜索，主线程结果为准。
pub fn search_par(env: &DarkChessEnv, cfg: &SearchConfig) -> Option<SearchResult> {
    let threads = cfg.threads.max(1);
    if threads == 1 {
        return search(env, cfg);
    }
    if let Some(nnue) = &cfg.nnue_evaluator {
        if !feature_dim_ok(env, &**nnue) {
            return None;
        }
    }
    let moves = env.generate_moves(env.get_current_player());
    if moves.is_empty() {
        return None;
    }
    let shared = Arc::new(SharedTT::new(cfg.tt_bits));
    let stop = Arc::new(AtomicBool::new(false));

    std::thread::scope(|scope| {
        for _ in 1..threads {
            let shared = Arc::clone(&shared);
            let stop = Arc::clone(&stop);
            let helper_env = *env;
            let helper_cfg = cfg.clone();
            scope.spawn(move || {
                // 助线程独立迭代加深，仅通过共享 TT 贡献信息，结果被丢弃。
                let budget = (helper_cfg.node_budget / threads as u64).max(1024);
                let mut helper_cfg = helper_cfg;
                helper_cfg.node_budget = budget;
                let Some(nnue) = helper_cfg.nnue_evaluator.clone() else {
                    return;
                };
                let root_acc = nnue.init_accumulator(&helper_env);
                let mut ctx = Ctx::with_tt(&helper_cfg, &helper_env, shared);
                if helper_cfg.feat(FEAT_REP) {
                    ctx.path.push(zobrist::zkey(&helper_env));
                }
                let _ = iterative_deepen(&helper_env, &*root_acc, &helper_cfg, &mut ctx, None, Some(&stop));
            });
        }
        let result = search_with_tt(env, cfg, Arc::clone(&shared));
        stop.store(true, Ordering::Relaxed);
        result
    })
}

/// 以指定共享置换表执行迭代加深主搜索。
fn search_with_tt(
    env: &DarkChessEnv,
    cfg: &SearchConfig,
    shared: Arc<SharedTT>,
) -> Option<SearchResult> {
    let nnue = require_nnue(
        env,
        cfg,
        "❌ Expectimax 搜索需要 NNUE 评估器：请经 SearchConfig.nnue_evaluator 注入（当前为 None）",
    )?;
    let moves = env.generate_moves(env.get_current_player());
    if moves.is_empty() {
        return None;
    }
    let root_acc = nnue.init_accumulator(env);
    let mut ctx = Ctx::with_tt(cfg, env, shared);
    if cfg.feat(FEAT_REP) {
        ctx.path.push(zobrist::zkey(env));
    }
    let (_, best_action, best_score, depth_reached) =
        iterative_deepen(env, &*root_acc, cfg, &mut ctx, None, None);
    let best = best_action.unwrap_or(moves[0].action);
    if cfg.tt_sym_probe && cfg.feat(FEAT_TT) {
        let [misses, raw_hits, sym_hits, sym_deep] = ctx.tt.tt_stats();
        let rate = if misses > 0 { sym_hits as f64 / misses as f64 } else { 0.0 };
        eprintln!(
            "[tt-sym-quant] 探测节点={} 原始命中={raw_hits} miss={misses} 对称命中={sym_hits} 对称命中(深度足够)={sym_deep} miss挽回率={rate:.1}%",
            misses + raw_hits
        );
    }
    Some(SearchResult {
        action: best,
        value: best_score,
        depth: depth_reached,
        nodes: ctx.nodes,
    })
}
