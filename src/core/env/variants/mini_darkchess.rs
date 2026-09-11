// ==============================================================================
// --- 4x2 迷你暗棋环境 (MiniDarkChessEnv) ---
//
// 复用共享的 DarkChessEnv 核心逻辑（config 驱动），仅以 mini_config() 区分：
// - 棋盘 4x2（8 格）
// - 仅 兵 / 炮 / 士 / 将 四种棋子，每方各 1 子（共 8 子填满棋盘）
// - 血量上限 = 2 + 5 + 10 + 30 = 47（= 单方棋子价值总和），全灭敌方即判胜
//
// 关联常量与 GameEnv 委托实现由 variants::impl_darkchess_variant! 宏统一生成。
// ==============================================================================

use crate::core::env::board::DarkChessEnv;
use crate::core::env::config::mini_config;

#[derive(Clone, Copy, Debug)]
pub struct MiniDarkChessEnv {
    pub inner: DarkChessEnv,
}

impl_darkchess_variant!(
    MiniDarkChessEnv,
    mini_config,
    MINI_ACTION_SPACE_SIZE,
    MINI_RESNET_BOARD_CHANNELS,
    MINI_BOARD_ROWS,
    MINI_BOARD_COLS,
    MINI_RESNET_SCALAR_FEATURE_COUNT,
);
