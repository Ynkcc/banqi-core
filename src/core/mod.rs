//! 领域核心模块 (Domain Core)
//!
//! 包含暗棋游戏环境逻辑 (`env`，含结构化走子生成)、
//! 通用 Gumbel MCTS 树搜索算法 (`mcts`) 与 Expectimax 强搜索引擎 (`expectimax`)。

pub mod env;
pub mod expectimax;
pub mod mcts;

