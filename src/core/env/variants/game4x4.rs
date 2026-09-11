// ==============================================================================
// --- 4x4 暗棋环境 (Game4x4Env) ---
//
// 复用共享的 DarkChessEnv 核心逻辑（config 驱动），仅以 game_4x4_config() 区分：
// - 棋盘 4x4（16 格）
// - 7 类棋子全激活，每方：兵2 炮1 马1 车1 象1 士1 将1（共 8 子填满棋盘）
// - 分值：兵4 / 炮10 / 马10 / 车10 / 象10 / 士20 / 将30
// - 血量上限 = 60（由变体指定，独立于分值总和）
//
// 关联常量与 GameEnv 委托实现由 variants::impl_darkchess_variant! 宏统一生成。
// ==============================================================================

use crate::core::env::board::DarkChessEnv;
use crate::core::env::config::game_4x4_config;

#[derive(Clone, Copy, Debug)]
pub struct Game4x4Env {
    pub inner: DarkChessEnv,
}

impl_darkchess_variant!(
    Game4x4Env,
    game_4x4_config,
    GAME4X4_ACTION_SPACE_SIZE,
    GAME4X4_RESNET_BOARD_CHANNELS,
    GAME4X4_BOARD_ROWS,
    GAME4X4_BOARD_COLS,
    GAME4X4_RESNET_SCALAR_FEATURE_COUNT,
);
