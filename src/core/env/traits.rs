// src/game_env/traits.rs
// 泛型游戏环境抽象：Gumbel MCTS 依赖的最小接口集。
//
// 背景：
// - 本项目最初只支持暗棋（DarkChessEnv），MCTS（mcts/）被硬编码绑定到
//   具体的 DarkChessEnv / Slot / Piece 类型上。
// - 为了复用同一套搜索核心做井字棋（Tic-Tac-Toe）与 4x2 迷你暗棋验证，
//   这里抽出 `GameEnv` trait。
// - 机会节点（翻牌随机性）是暗棋特有语义：trait 提供 `is_chance_action` /
//   `chance_outcomes` / `step_outcome_id` 三个扩展点，默认实现为「无机会节点」，
//   DarkChessEnv 与 MiniDarkChessEnv 覆盖实现、TicTacToeEnv 保持默认。

use super::board::DarkChessEnv;
use super::config::GameConfig;
use super::types::{ResNetObservation, Piece, Player};

/// 泛型游戏环境：Gumbel MCTS 对其施加的全部约束。
///
/// 要求 `Copy`：MCTS 节点以值语义保存环境快照（与既有 DarkChessEnv 的 Copy 设计一致）。
pub trait GameEnv: Copy + Clone + Send + Sync + 'static {
    /// 动作空间大小（运行时值：由 `config` 决定，4x8 / 4x4 / 4x2 各不相同）
    fn action_space_size(&self) -> usize;

    /// 当前玩家
    fn get_current_player(&self) -> Player;

    /// 将合法动作掩码写入 `masks`（合法位置 1，其余位置 0）。
    /// 调用方保证 `masks.len() >= self.action_space_size()`。
    fn action_masks_into(&self, masks: &mut [i32]);

    /// 执行动作，返回 `(奖励, 是否终止, 是否截断, 胜者)`。
    ///
    /// 观测不随步返回，按需调用 `get_resnet_state()`。
    /// `winner` 使用全局视角：`Some(1)` = 红方/先手胜，`Some(-1)` = 黑方/后手胜，
    /// `Some(0)` = 平局，`None` = 未结束。
    fn step(&mut self, action: usize) -> Result<(f32, bool, bool, Option<i32>), String>;

    /// 获取当前观测（神经网络输入）。
    ///
    /// 形状不能由编译期常量表达：`DarkChessEnv` 的棋盘行列 / 通道 / 标量数
    /// 由运行时 `config` 决定，同一类型可对应 4x8 / 4x4 / 4x2 三种变体。
    fn get_resnet_state(&self) -> ResNetObservation;

    /// 终局检测：`(terminated, truncated, winner)`
    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>);

    /// 每局最大步数（步数上限截断）。
    fn max_steps(&self) -> usize;

    // ------------------------------------------------------------------------
    // 神经网络特征（供批量推理 / Python 绑定使用）
    // ------------------------------------------------------------------------

    /// 将环境编码为扁平特征写入外部缓冲区。
    fn encode_resnet_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>);

    // ------------------------------------------------------------------------
    // 机会节点扩展点
    // ------------------------------------------------------------------------

    fn is_chance_action(&self, _action: usize) -> bool {
        false
    }

    fn chance_outcomes(&self, _action: usize) -> Vec<(usize, f32, Self)> {
        Vec::new()
    }

    fn step_outcome_id(&self, _action: usize) -> Option<usize> {
        None
    }

    // ------------------------------------------------------------------------
    // 终局血量差（训练/归档辅助数据）
    // ------------------------------------------------------------------------

    /// 终局归一化血量差（红方视角为正）。
    ///
    /// 公式：`(红方HP - 黑方HP) / (初始总HP + 最大子力分值)`，大致落在 [-1, 1]。
    /// 在终局（与获取游戏真实结果同一时机）调用。无血量机制的游戏（如井字棋）返回 None。
    fn terminal_health_diff_red(&self) -> Option<f32> {
        None
    }

    /// 终局整型血量差（红方视角：红HP - 黑HP，未归一化）。
    ///
    /// 供 MCTS 终局分支计算血量期望（分桶中心为整型差，D = initial_health）。
    /// 无血量机制的游戏返回 None。
    fn terminal_health_diff_red_int(&self) -> Option<i32> {
        None
    }

    /// 血量差的归一化标尺（= initial_health，D），把整型差映射到 [-1, 1]。
    ///
    /// 与离散分类头分桶定义一致（D = (K-1)/2）。无血量机制返回 0。
    fn health_diff_scale(&self) -> f32 {
        0.0
    }
}

// ============================================================================
// 暗棋实现
// ============================================================================

/// 获取棋子的唯一结果 ID（暗棋机会节点的可能结果标识）。
///
/// ID 计算方式：按 config 的激活类型紧凑索引 + 玩家偏移（红方 0，黑方 num_active）。
pub fn get_outcome_id(cfg: &GameConfig, piece: &Piece) -> usize {
    cfg.outcome_id_for(piece.piece_type, piece.player == Player::Black)
}

impl GameEnv for DarkChessEnv {
    fn action_space_size(&self) -> usize {
        self.config.action_space_size
    }

    fn get_current_player(&self) -> Player {
        DarkChessEnv::get_current_player(self)
    }

    fn action_masks_into(&self, masks: &mut [i32]) {
        DarkChessEnv::action_masks_into(self, masks);
    }

    fn step(&mut self, action: usize) -> Result<(f32, bool, bool, Option<i32>), String> {
        DarkChessEnv::step(self, action, None)
    }

    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        DarkChessEnv::check_game_over_conditions(self)
    }

    fn max_steps(&self) -> usize {
        self.config.max_steps_per_episode
    }

    fn encode_resnet_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        DarkChessEnv::resnet_features_flat_into(self, board_data, scalars_data);
    }

    fn get_resnet_state(&self) -> ResNetObservation {
        DarkChessEnv::get_resnet_state(self)
    }

    // --- 机会节点 ---

    fn is_chance_action(&self, action: usize) -> bool {
        DarkChessEnv::is_chance_action(self, action)
    }

    fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
        DarkChessEnv::chance_outcomes(self, action)
    }

    fn step_outcome_id(&self, action: usize) -> Option<usize> {
        DarkChessEnv::step_outcome_id(self, action)
    }

    fn terminal_health_diff_red(&self) -> Option<f32> {
        let denom = self.config.initial_health as f32
            + self.config.piece_values.iter().copied().max().unwrap_or(0) as f32;
        if denom <= 0.0 {
            None
        } else {
            Some((self.get_hp(Player::Red) - self.get_hp(Player::Black)) as f32 / denom)
        }
    }

    fn terminal_health_diff_red_int(&self) -> Option<i32> {
        Some(self.get_hp(Player::Red) as i32 - self.get_hp(Player::Black) as i32)
    }

    fn health_diff_scale(&self) -> f32 {
        self.config.initial_health as f32
    }
}

// 4x4 暗棋 / 4x2 迷你暗棋的 `GameEnv` 委托实现
// 由 `variants::impl_darkchess_variant!` 宏统一生成（见 variants/mod.rs）。
