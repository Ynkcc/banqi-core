//! Expectimax 搜索配置与结果类型。

use std::sync::Arc;

use super::nnue::NnueEvaluate;
use super::{FEAT_LMR, FEAT_ORDERING, FEAT_REP, FEAT_TT};

/// 搜索引擎配置
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// 节点预算（总节点数上限；超出即中止当前迭代）
    pub node_budget: u64,
    /// 时间预算（毫秒；0 = 仅按节点预算）
    pub time_limit_ms: u64,
    /// 迭代加深最大深度
    pub max_depth: i32,
    /// 和棋偏差（contempt；正数 = 领先方避和、落后方求和）
    pub contempt: f32,
    /// 是否启用静态搜索
    pub quiesce: bool,
    /// 静态搜索最大深度
    pub quiesce_max: i32,
    /// 特性位掩码（FEAT_*）
    pub features: u32,
    /// 置换表大小（2^tt_bits 项）
    pub tt_bits: u32,
    /// Lazy SMP 并发线程数（1 = 单线程，>1 时共享置换表协同搜索）
    pub threads: usize,
    /// 机会节点子分支额外减深（翻棋信息价值低，0 = 不减）
    pub chance_reduction: i32,
    /// 量化开关：TT 原始键 miss 后用对称视角键二次探测并统计（只计数，不参与存储/截断）
    pub tt_sym_probe: bool,
    /// NNUE 求值网络引擎（叶评估唯一来源；未加载时 `search` 拒绝执行）
    pub nnue_evaluator: Option<Arc<dyn NnueEvaluate>>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            node_budget: 500_000,
            time_limit_ms: 0,
            max_depth: 24,
            contempt: 0.1,
            quiesce: true,
            quiesce_max: 8,
            features: FEAT_ORDERING | FEAT_TT | FEAT_LMR | FEAT_REP,
            tt_bits: 18,
            threads: 1,
            chance_reduction: 1,
            tt_sym_probe: false,
            nnue_evaluator: None,
        }
    }
}

impl SearchConfig {
    #[inline]
    pub(super) fn feat(&self, bit: u32) -> bool {
        self.features & bit != 0
    }
}

/// 搜索引擎评估与搜索结果
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub action: usize,
    /// 根走子方视角的评估值（最深完成迭代）
    pub value: f32,
    /// 完成的最深迭代层数
    pub depth: i32,
    /// 消耗的总节点数
    pub nodes: u64,
}
