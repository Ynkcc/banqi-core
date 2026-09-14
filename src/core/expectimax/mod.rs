//! # Expectimax 核心搜索引擎 (参照 core/mcts 风格立为 core 一级模块)
//!
//! 包含 Star1 概率节点剪枝 Expecti-Alpha-Beta 强搜索核心 `ExpectimaxEngine`
//! （置换表 + 走子排序 + 静态搜索 + LMR + 迭代加深），叶评估以 NNUE 为唯一来源。
//!
//! 子模块分层：
//!   - search:    搜索入口与 Lazy SMP（search / search_par）
//!   - negamax:   递归主体（negamax + Star1 机会节点 + quiescence + LMR + TT）
//!   - iterative: 迭代加深（根层单层搜索 + 逐层加深）
//!   - eval:      叶评估与增量累加器辅助
//!   - config:    SearchConfig / SearchResult（经 search 再导出）
//!   - ordering:  走子排序（MVV-LVA + 杀手 + 历史）+ 终局检测/价值
//!   - nnue:      NNUE 叶评估抽象（trait 契约，Expectimax 唯一叶评估来源）
//!   - zobrist:   Zobrist 局面哈希 + 值域常量 + 置换表 TtEntry
//!
//! 值约定：所有搜索值均为“当前节点走子方视角”，范围约 [-1, 1]。

use std::sync::Arc;
use std::time::Instant;

use crate::core::env::DarkChessEnv;
use crate::core::env::Move;
use nnue::NnueEvaluate;

pub mod nnue;
pub mod ordering;
pub mod search;
pub mod smp;
pub mod zobrist;

mod config;
mod eval;
mod iterative;
mod negamax;

#[cfg(test)]
mod tests;

pub use search::{SearchConfig, SearchResult, search, search_par};

pub use smp::SharedTT;

// 搜索特性标志位
pub const FEAT_ORDERING: u32 = 1 << 0; // 走子排序（MVV-LVA + 杀手 + 历史）
pub const FEAT_TT: u32 = 1 << 1; // 置换表（决策节点）
pub const FEAT_LMR: u32 = 1 << 2; // 晚走子减深（late move reductions）
pub const FEAT_REP: u32 = 1 << 3; // 重复局面检测（路径 zkey）

/// 搜索上下文。
///
/// `tt` 为共享置换表句柄：单线程搜索时独占一个实例；Lazy SMP 多线程时由
/// 主/助线程共享同一 `Arc` 交换信息。
pub struct Ctx {
    nodes: u64,
    budget: u64,
    start: Instant,
    time_limit_ms: u64,
    killers: Vec<[usize; 2]>, // 每个剩余深度的两个杀手动作
    history: Vec<i32>,        // [total_positions*total_positions] 静走子截断历史
    tt: Arc<smp::SharedTT>,
    path: Vec<u64>, // 当前搜索路径上的祖先 zkey（重复检测）
    root: usize,    // 根走子方 idx（contempt 方向）
}

impl Ctx {
    fn with_tt(cfg: &SearchConfig, env: &DarkChessEnv, tt: Arc<smp::SharedTT>) -> Self {
        let total = env.config.total_positions;
        let kd = (cfg.max_depth.max(1) + 2) as usize;
        Self {
            nodes: 0,
            budget: cfg.node_budget.max(1),
            start: Instant::now(),
            time_limit_ms: cfg.time_limit_ms,
            killers: vec![[0; 2]; kd],
            history: vec![0; total * total],
            tt,
            path: Vec::with_capacity(64),
            root: env.get_current_player().idx(),
        }
    }

    #[inline]
    fn tick(&mut self) -> Result<(), ()> {
        self.nodes += 1;
        if self.nodes > self.budget {
            return Err(());
        }
        if self.time_limit_ms > 0
            && (self.nodes & 1023) == 0
            && self.start.elapsed().as_millis() as u64 >= self.time_limit_ms
        {
            return Err(());
        }
        Ok(())
    }

    /// 记录静走子截断：提升为杀手走并累加历史分（深度²）。
    #[inline]
    fn record_cutoff(&mut self, m: &Move, depth: i32, total: usize) {
        let d = depth as usize;
        if d < self.killers.len() && self.killers[d][0] != m.action {
            self.killers[d][1] = self.killers[d][0];
            self.killers[d][0] = m.action;
        }
        let key = m.from * total + m.to;
        if key < self.history.len() {
            self.history[key] += depth * depth;
        }
    }
}

/// Expectimax 独立搜索引擎实体
pub struct ExpectimaxEngine {
    pub config: SearchConfig,
}

impl ExpectimaxEngine {
    /// 以指定 NNUE 评估器创建引擎（叶评估以 NNUE 为唯一来源，搜索强制要求）
    pub fn with_nnue(evaluator: Arc<dyn NnueEvaluate>) -> Self {
        let mut config = SearchConfig::default();
        config.nnue_evaluator = Some(evaluator);
        Self { config }
    }

    /// 设置搜索最大深度
    pub fn set_max_depth(&mut self, depth: i32) {
        self.config.max_depth = depth;
    }

    /// 设置节点预算
    pub fn set_node_budget(&mut self, budget: u64) {
        self.config.node_budget = budget;
    }

    /// 搜寻最佳走子
    pub fn search(&self, env: &DarkChessEnv) -> Option<SearchResult> {
        search(env, &self.config)
    }

    /// 搜寻最佳走子（Lazy SMP 多线程；`config.threads <= 1` 时等价 `search`）
    pub fn search_par(&self, env: &DarkChessEnv) -> Option<SearchResult> {
        search_par(env, &self.config)
    }

    /// 搜寻最佳动作编号
    pub fn best_action(&self, env: &DarkChessEnv) -> Option<usize> {
        self.search(env).map(|res| res.action)
    }
}
