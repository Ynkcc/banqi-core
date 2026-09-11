//! NNUE 叶评估抽象（trait）。
//!
//! Expectimax 以 NNUE 为唯一叶评估来源；本模块只定义评估器与增量累加器的
//! 接口契约，具体量化网络实现（`NnueEvaluator`/`NnueBoard`）在上层 crate
//! `banqi-engine` 中，经 trait 对象注入 `SearchConfig.nnue_evaluator`。

use std::sync::Arc;

use crate::core::env::types::Player;
use crate::core::env::DarkChessEnv;

/// NNUE 叶评估器（Expectimax 唯一叶评估来源）。
///
/// 实现方需 `Send + Sync + Debug`（SearchConfig 跨线程克隆与日志打印）。
pub trait NnueEvaluate: Send + Sync + std::fmt::Debug {
    /// 全量叶评估：当前行棋方视角，值域约 [-1, 1]。
    fn evaluate(&self, env: &DarkChessEnv) -> f32;

    /// 校验特征维度与环境推导维度一致（维度错位会导致评估静默失真，必须硬失败）。
    fn validate_feature_dim(&self, expected: usize) -> Result<(), String>;

    /// 从环境初始化红黑双视角累加器（供搜索 O(1) 增量评估）。
    ///
    /// `self: Arc<Self>` 使实现方在构造累加器时可捕获自身的 Arc 引用
    /// （累加器增量更新需要权重数据，且必须独立于 `&self` 借用存活）。
    fn init_accumulator(self: Arc<Self>, env: &DarkChessEnv) -> Box<dyn NnueAccumulator>;
}

/// 双视角特征累加器句柄（由 `NnueEvaluate::init_accumulator` 产出）。
pub trait NnueAccumulator: Send + Sync {
    /// 深拷贝：搜索为每个子节点复制一份累加器。
    fn clone_box(&self) -> Box<dyn NnueAccumulator>;

    /// 一步动作后的增量更新（实现方自行计算红黑双方视角的特征差分）。
    fn apply_step(&mut self, before: &DarkChessEnv, after: &DarkChessEnv, action: usize);

    /// 指定视角的 O(1) 快速评估（当前累加器状态，值域约 [-1, 1]）。
    fn evaluate(&self, player: Player) -> f32;
}

#[cfg(test)]
pub(crate) mod test_support {
    //! 测试用零权重 Dummy 评估器（恒返回 0.0，仅验证搜索主流程）。

    use super::*;

    #[derive(Debug)]
    pub(crate) struct DummyNnue;

    impl NnueEvaluate for DummyNnue {
        fn evaluate(&self, _env: &DarkChessEnv) -> f32 {
            0.0
        }

        fn validate_feature_dim(&self, _expected: usize) -> Result<(), String> {
            Ok(())
        }

        fn init_accumulator(self: Arc<Self>, _env: &DarkChessEnv) -> Box<dyn NnueAccumulator> {
            Box::new(DummyAcc)
        }
    }

    struct DummyAcc;

    impl NnueAccumulator for DummyAcc {
        fn clone_box(&self) -> Box<dyn NnueAccumulator> {
            Box::new(DummyAcc)
        }

        fn apply_step(&mut self, _before: &DarkChessEnv, _after: &DarkChessEnv, _action: usize) {}

        fn evaluate(&self, _player: Player) -> f32 {
            0.0
        }
    }
}
