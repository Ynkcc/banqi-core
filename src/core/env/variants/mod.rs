//! 游戏变体环境子模块
//! 包含 4x4 暗棋、4x2 迷你暗棋及井字棋环境。

use crate::core::env::board::DarkChessEnv;

pub mod game4x4;
pub mod mini_darkchess;
pub mod tic_tac_toe;

pub use game4x4::{
    GAME4X4_ACTION_SPACE_SIZE, GAME4X4_RESNET_BOARD_CHANNELS, GAME4X4_BOARD_COLS, GAME4X4_BOARD_ROWS,
    GAME4X4_RESNET_SCALAR_FEATURE_COUNT, Game4x4Env,
};
pub use mini_darkchess::{
    MINI_ACTION_SPACE_SIZE, MINI_RESNET_BOARD_CHANNELS, MINI_BOARD_COLS, MINI_BOARD_ROWS,
    MINI_RESNET_SCALAR_FEATURE_COUNT, MiniDarkChessEnv,
};
pub use tic_tac_toe::{
    TTT_ACTION_SPACE_SIZE, TTT_RESNET_BOARD_CHANNELS, TTT_BOARD_COLS, TTT_BOARD_ROWS,
    TTT_RESNET_SCALAR_FEATURE_COUNT, TicTacToeEnv,
};

/// 支持覆盖初始预翻棋子数的暗棋变体环境（课程学习用）。
///
/// 仅改变 `GameConfig.initial_revealed_pieces`，棋盘尺寸 / 动作空间 /
/// 特征维度均不变，因此各阶段可复用同一网络结构。
pub trait CurriculumEnv: Sized + Default {
    /// 以指定初始翻子数创建环境（n 超过格数时由环境内部截断）。
    fn with_initial_revealed(n: usize) -> Self;
}

impl CurriculumEnv for DarkChessEnv {
    fn with_initial_revealed(n: usize) -> Self {
        let mut cfg = crate::core::env::config::darkchess_config();
        cfg.initial_revealed_pieces = n;
        Self::with_config(cfg)
    }
}

impl CurriculumEnv for Game4x4Env {
    fn with_initial_revealed(n: usize) -> Self {
        let mut cfg = crate::core::env::config::game_4x4_config();
        cfg.initial_revealed_pieces = n;
        Self {
            inner: DarkChessEnv::with_config(cfg),
        }
    }
}

impl CurriculumEnv for MiniDarkChessEnv {
    fn with_initial_revealed(n: usize) -> Self {
        let mut cfg = crate::core::env::config::mini_config();
        cfg.initial_revealed_pieces = n;
        Self {
            inner: DarkChessEnv::with_config(cfg),
        }
    }
}
