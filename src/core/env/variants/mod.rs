//! 游戏变体环境子模块
//! 包含 4x4 暗棋、4x2 迷你暗棋及井字棋环境。

/// 暗棋变体统一实现宏：为「包装 DarkChessEnv 的变体环境」生成
/// 模块级特征常量（单一真源为 config 函数）、固有委托方法、`Default`
/// 以及 `GameEnv` 委托实现。
///
/// 变体只需定义 `struct X { inner: DarkChessEnv }` 并调用本宏。
macro_rules! impl_darkchess_variant {
    (
        $ty:ident, $cfg:ident,
        $ACT:ident, $CH:ident, $ROWS:ident, $COLS:ident, $SCALARS:ident $(,)?
    ) => {
        /// 动作空间大小（由 config 推导）。
        pub const $ACT: usize = $cfg().action_space_size;
        /// 棋盘特征通道数 = 2 * num_active + 2（由 config 推导）。
        pub const $CH: usize = $cfg().resnet_board_channels;
        /// 棋盘行数。
        pub const $ROWS: usize = $cfg().rows;
        /// 棋盘列数。
        pub const $COLS: usize = $cfg().cols;
        /// 标量特征数（由 config 推导）。
        pub const $SCALARS: usize = $cfg().resnet_scalar_feature_count;

        impl $ty {
            pub fn new() -> Self {
                Self {
                    inner: DarkChessEnv::with_config($cfg()),
                }
            }

            pub fn action_space_size(&self) -> usize {
                $cfg().action_space_size
            }

            pub fn max_steps(&self) -> usize {
                $cfg().max_steps_per_episode
            }

            pub fn get_current_player(&self) -> crate::core::env::types::Player {
                self.inner.get_current_player()
            }

            /// 切换当前玩家（不改变棋盘/棋子归属，仅改变编码视角）。
            pub fn flip_player(&mut self) {
                self.inner.flip_player();
            }

            pub fn action_masks_into(&self, masks: &mut [i32]) {
                self.inner.action_masks_into(masks);
            }

            pub fn step(
                &mut self,
                action: usize,
            ) -> Result<(f32, bool, bool, Option<i32>), String> {
                self.inner.step(action, None)
            }

            pub fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
                self.inner.check_game_over_conditions()
            }

            pub fn encode_resnet_features_flat_into(
                &self,
                board_data: &mut Vec<f32>,
                scalars_data: &mut Vec<f32>,
            ) {
                self.inner.resnet_features_flat_into(board_data, scalars_data);
            }

            pub fn is_chance_action(&self, action: usize) -> bool {
                self.inner.is_chance_action(action)
            }

            pub fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
                self.inner
                    .chance_outcomes(action)
                    .into_iter()
                    .map(|(id, p, env)| (id, p, Self { inner: env }))
                    .collect()
            }

            pub fn step_outcome_id(&self, action: usize) -> Option<usize> {
                self.inner.step_outcome_id(action)
            }

            pub fn nnue_active_features(&self) -> Vec<usize> {
                self.inner.nnue_active_features()
            }

            /// 打印棋盘（方便演示/调试）。
            pub fn print_board(&self) {
                self.inner.print_board();
            }
        }

        impl Default for $ty {
            fn default() -> Self {
                Self::new()
            }
        }

        impl crate::core::env::GameEnv for $ty {
            fn action_space_size(&self) -> usize {
                $ACT
            }

            fn get_current_player(&self) -> crate::core::env::types::Player {
                self.inner.get_current_player()
            }

            fn action_masks_into(&self, masks: &mut [i32]) {
                self.inner.action_masks_into(masks);
            }

            fn step(&mut self, action: usize) -> Result<(f32, bool, bool, Option<i32>), String> {
                self.inner.step(action, None)
            }

            fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
                self.inner.check_game_over_conditions()
            }

            fn max_steps(&self) -> usize {
                $cfg().max_steps_per_episode
            }

            fn encode_resnet_features_flat_into(
                &self,
                board_data: &mut Vec<f32>,
                scalars_data: &mut Vec<f32>,
            ) {
                self.inner.resnet_features_flat_into(board_data, scalars_data);
            }

            fn get_resnet_state(&self) -> crate::core::env::types::ResNetObservation {
                self.inner.get_resnet_state()
            }

            fn is_chance_action(&self, action: usize) -> bool {
                self.inner.is_chance_action(action)
            }

            fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
                self.inner
                    .chance_outcomes(action)
                    .into_iter()
                    .map(|(id, p, env)| (id, p, Self { inner: env }))
                    .collect()
            }

            fn step_outcome_id(&self, action: usize) -> Option<usize> {
                self.inner.step_outcome_id(action)
            }

            fn terminal_health_diff_red(&self) -> Option<f32> {
                self.inner.terminal_health_diff_red()
            }

            fn terminal_health_diff_red_int(&self) -> Option<i32> {
                self.inner.terminal_health_diff_red_int()
            }

            fn health_diff_scale(&self) -> f32 {
                self.inner.health_diff_scale()
            }
        }
    };
}

use crate::core::env::board::DarkChessEnv;

pub mod game4x4;
pub mod mini_darkchess;
pub mod tic_tac_toe;

pub use game4x4::{
    GAME4X4_ACTION_SPACE_SIZE, GAME4X4_BOARD_COLS, GAME4X4_BOARD_ROWS,
    GAME4X4_RESNET_BOARD_CHANNELS, GAME4X4_RESNET_SCALAR_FEATURE_COUNT, Game4x4Env,
};
pub use mini_darkchess::{
    MINI_ACTION_SPACE_SIZE, MINI_BOARD_COLS, MINI_BOARD_ROWS, MINI_RESNET_BOARD_CHANNELS,
    MINI_RESNET_SCALAR_FEATURE_COUNT, MiniDarkChessEnv,
};
pub use tic_tac_toe::{
    TTT_ACTION_SPACE_SIZE, TTT_BOARD_COLS, TTT_BOARD_ROWS, TTT_RESNET_BOARD_CHANNELS,
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
