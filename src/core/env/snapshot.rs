// src/core/env/snapshot.rs
// 局面快照：把暗棋状态编码成紧凑字节串，供跨进程 / 跨机器的「局面重搜（reanalysis）」使用。
//
// 为什么不能复用已有入口：
//   - episode 里的特征（board / scalars）只含**观测**信息（暗子只有「此格有暗子」+ 按型计数），
//     无法还原出可搜索的完整状态；
//   - `DarkChessEnv::from_board` 只还原棋盘与行棋方，步数计数 / 血量 / 暗子袋 / 阵亡计数
//     全部复位，直接拿它重搜会得到语义不同的目标（判和步数、血量、翻子概率都变了）。
//
// 不变量：快照 + 变体配置可**精确**重建搜索语义相关的全部状态（见 `from_snapshot`）。
// 未还原的只有两处，均不影响搜索语义：
//   - `last_revealed_piece`：只在 `step_next` 内部消费的瞬时字段（重搜从新树开始，不经过它）；
//   - `dead_pieces_pool` 的插入顺序：仅影响 `get_dead_pieces()` 的展示顺序，语义只用计数
//     （全灭判定用 `dead_pieces_count`，特征用 `dead_piece_counts_by_type`）。
//
// 随机性：未设种子的对局在还原后翻子是**重新随机抽取**的（原局同样是随机抽取），
// 搜索语义一致；设种子的对局会连 `true_board` 一起快照，保障轨迹可复现。

use super::board::DarkChessEnv;
use super::config::{MAX_PIECES_PER_PLAYER, MAX_POSITIONS, NUM_PIECE_TYPES_MAX, Variant};
use super::error::EnvError;
use super::types::{Piece, PieceType, Player, Slot};
use super::bitboard::ull;

/// 快照格式版本：布局变更时递增，解码端据此拒绝旧/新格式（不做兼容猜测）。
const SNAPSHOT_VERSION: u8 = 1;

/// 一个局面的完整快照（可编码为字节串跨进程传输）。
#[derive(Clone, Debug, PartialEq)]
pub struct PositionSnapshot {
    /// 变体（解码时据此取 `GameConfig`）。
    pub variant: Variant,
    /// 棋盘槽位，长度 = `config.total_positions`。
    pub slots: Vec<Slot>,
    /// 行棋方。
    pub current_player: Player,
    /// 连续无吃子步数（判和计数）。
    pub move_counter: u32,
    /// 游戏总步数（步数上限截断）。
    pub total_step_counter: u32,
    /// 双方血量。
    pub scores: [i32; 2],
    /// 上一步动作（初始局面为 -1）。
    pub last_action: i32,
    /// 暗子袋（未翻开的棋子池；顺序与原始状态一致，`reveal_probabilities` 由它推导）。
    pub hidden_bag: Vec<Piece>,
    /// 按棋子类型的阵亡计数 `[PlayerIdx][PieceType]`（存活向量与全灭判定使用）。
    pub dead_counts: [[u8; NUM_PIECE_TYPES_MAX]; 2],
    /// 预置真实布局：仅设种子的对局有（长度 = `config.total_positions`）。
    pub true_board: Option<Vec<Piece>>,
}

// ============================================================================
// 编码 / 解码（小端、定长字段，解码全程边界检查，非法输入返回 Err 而非 panic）
// ============================================================================

impl PositionSnapshot {
    /// 编码为字节串（版本号 + 变体下标 + 状态字段）。
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(160);
        out.push(SNAPSHOT_VERSION);
        out.push(self.variant.index() as u8);
        out.extend(self.slots.iter().map(encode_slot));
        out.push(self.current_player.idx() as u8);
        out.extend_from_slice(&self.move_counter.to_le_bytes());
        out.extend_from_slice(&self.total_step_counter.to_le_bytes());
        out.extend_from_slice(&self.scores[0].to_le_bytes());
        out.extend_from_slice(&self.scores[1].to_le_bytes());
        out.extend_from_slice(&self.last_action.to_le_bytes());
        out.push(self.hidden_bag.len() as u8);
        out.extend(self.hidden_bag.iter().map(encode_piece));
        for counts in &self.dead_counts {
            out.extend_from_slice(counts);
        }
        match &self.true_board {
            Some(tb) => {
                out.push(1);
                out.extend(tb.iter().map(encode_piece));
            }
            None => out.push(0),
        }
        out
    }

    /// 从字节串解码。任何结构性问题（版本不符 / 长度不足 / 编码非法 / 变体未知）都返回 Err。
    pub fn decode(bytes: &[u8]) -> Result<Self, EnvError> {
        let mut r = Reader { buf: bytes, pos: 0 };
        let version = r.u8()?;
        if version != SNAPSHOT_VERSION {
            return Err(EnvError::InvalidSnapshot { context: "快照版本不支持" });
        }
        let variant_idx = r.u8()? as usize;
        let variant = Variant::ALL
            .iter()
            .copied()
            .find(|v| v.index() == variant_idx)
            .ok_or(EnvError::InvalidSnapshot { context: "快照变体下标非法" })?;
        let n = variant.config().total_positions;

        let raw_slots = r.take(n)?;
        let mut slots = Vec::with_capacity(n);
        for &b in raw_slots {
            slots.push(decode_slot(b)?);
        }

        let current_player = match r.u8()? {
            0 => Player::Red,
            1 => Player::Black,
            _ => return Err(EnvError::InvalidSnapshot { context: "快照行棋方编码非法" }),
        };
        let move_counter = r.u32()?;
        let total_step_counter = r.u32()?;
        let scores = [r.i32()?, r.i32()?];
        let last_action = r.i32()?;

        let bag_len = r.u8()? as usize;
        if bag_len > MAX_POSITIONS {
            return Err(EnvError::InvalidSnapshot { context: "快照暗子袋长度越界" });
        }
        let mut hidden_bag = Vec::with_capacity(bag_len);
        for &b in r.take(bag_len)? {
            hidden_bag.push(decode_piece(b)?);
        }

        let mut dead_counts = [[0u8; NUM_PIECE_TYPES_MAX]; 2];
        for counts in dead_counts.iter_mut() {
            counts.copy_from_slice(r.take(NUM_PIECE_TYPES_MAX)?);
        }

        let true_board = match r.u8()? {
            0 => None,
            1 => {
                let mut tb = Vec::with_capacity(n);
                for &b in r.take(n)? {
                    tb.push(decode_piece(b)?);
                }
                Some(tb)
            }
            _ => return Err(EnvError::InvalidSnapshot { context: "快照 true_board 标记非法" }),
        };

        Ok(Self {
            variant,
            slots,
            current_player,
            move_counter,
            total_step_counter,
            scores,
            last_action,
            hidden_bag,
            dead_counts,
            true_board,
        })
    }
}

fn encode_piece(p: &Piece) -> u8 {
    (p.player.idx() as u8) * (NUM_PIECE_TYPES_MAX as u8) + (p.piece_type as usize as u8)
}

fn decode_piece(code: u8) -> Result<Piece, EnvError> {
    let player = match code / (NUM_PIECE_TYPES_MAX as u8) {
        0 => Player::Red,
        1 => Player::Black,
        _ => return Err(EnvError::InvalidSnapshot { context: "快照棋子编码非法" }),
    };
    Ok(Piece::new(PieceType::from_index((code % (NUM_PIECE_TYPES_MAX as u8)) as usize), player))
}

fn encode_slot(s: &Slot) -> u8 {
    match s {
        Slot::Empty => 0,
        Slot::Hidden => 1,
        Slot::Revealed(p) => 2 + encode_piece(p),
    }
}

fn decode_slot(b: u8) -> Result<Slot, EnvError> {
    match b {
        0 => Ok(Slot::Empty),
        1 => Ok(Slot::Hidden),
        _ => Ok(Slot::Revealed(decode_piece(b - 2)?)),
    }
}

/// 带边界的顺序读取器：解包越界一律返回 Err（不 panic）。
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], EnvError> {
        let end = self.pos.checked_add(n).ok_or(EnvError::InvalidSnapshot { context: "快照长度溢出" })?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or(EnvError::InvalidSnapshot { context: "快照长度不足" })?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, EnvError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, EnvError> {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(b))
    }

    fn i32(&mut self) -> Result<i32, EnvError> {
        let mut b = [0u8; 4];
        b.copy_from_slice(self.take(4)?);
        Ok(i32::from_le_bytes(b))
    }
}

// ============================================================================
// DarkChessEnv ↔ 快照
// ============================================================================

/// 可导出 / 重建局面快照的环境（`DarkChessEnv` 与三个变体包装均实现）。
///
/// 泛型代码（如跨进程重搜流程）只依赖本 trait，不必知道具体盘面类型；
/// 变体包装的实现会校验「快照变体 == 本变体」，避免把 4x2 的局面当 4x8 重搜。
pub trait SnapshotEnv: Sized {
    /// 导出当前局面的完整快照。
    fn to_snapshot(&self) -> PositionSnapshot;
    /// 由快照精确重建环境（载荷非法 / 变体不符时返回 Err）。
    fn from_snapshot(s: &PositionSnapshot) -> Result<Self, EnvError>;
}

impl SnapshotEnv for DarkChessEnv {
    /// 导出当前局面的完整快照。
    fn to_snapshot(&self) -> PositionSnapshot {
        let n = self.config.total_positions;
        PositionSnapshot {
            variant: self.config.variant,
            slots: self.board[..n].to_vec(),
            current_player: self.current_player,
            move_counter: self.move_counter as u32,
            total_step_counter: self.total_step_counter as u32,
            scores: self.scores,
            last_action: self.last_action,
            hidden_bag: self.hidden_pieces_pool[..self.hidden_pieces_count].to_vec(),
            dead_counts: self.dead_piece_counts_by_type,
            true_board: self.true_board.map(|tb| tb[..n].to_vec()),
        }
    }

    /// 由快照精确重建环境（位棋盘 / 暗子袋 / 揭示概率 / 阵亡计数 / 标量状态全部还原）。
    ///
    /// 结构性非法（槽位数与变体不符 / 阵亡计数超出上限 / 步数越界）返回 Err：
    /// 这类输入只可能来自损坏或跨版本的载荷，静默接受会产出语义错误的重搜目标。
    fn from_snapshot(s: &PositionSnapshot) -> Result<Self, EnvError> {
        let config = s.variant.config();
        let n = config.total_positions;
        if s.slots.len() != n {
            return Err(EnvError::InvalidSnapshot { context: "快照槽位数与变体不符" });
        }
        if let Some(tb) = &s.true_board {
            if tb.len() != n {
                return Err(EnvError::InvalidSnapshot { context: "快照 true_board 长度与变体不符" });
            }
        }
        if s.hidden_bag.len() > MAX_POSITIONS {
            return Err(EnvError::InvalidSnapshot { context: "快照暗子袋长度越界" });
        }
        if s.move_counter as usize > config.max_steps_per_episode
            || s.total_step_counter as usize > config.max_steps_per_episode
        {
            return Err(EnvError::InvalidSnapshot { context: "快照步数超出对局上限" });
        }

        let mut board = [Slot::Empty; MAX_POSITIONS];
        board[..n].copy_from_slice(&s.slots);
        let mut tb_arr = [Piece::default(); MAX_POSITIONS];
        let true_board = match &s.true_board {
            Some(tb) => {
                tb_arr[..n].copy_from_slice(tb);
                Some(tb_arr)
            }
            None => None,
        };

        let mut env = Self::fresh_with(config, board, None, true_board);

        // 位棋盘：由槽位重建（Empty / Hidden / Revealed 互斥完备）
        for sq in 0..n {
            let mask = ull(sq);
            match env.board[sq] {
                Slot::Empty => env.empty_bitboard |= mask,
                Slot::Hidden => env.hidden_bitboard |= mask,
                Slot::Revealed(p) => {
                    env.revealed_bitboards[p.player.idx()] |= mask;
                    env.piece_bitboards[p.player.idx()][p.piece_type as usize] |= mask;
                }
            }
        }

        // 暗子袋 → 揭示概率表
        env.hidden_pieces_pool[..s.hidden_bag.len()].copy_from_slice(&s.hidden_bag);
        env.hidden_pieces_count = s.hidden_bag.len();
        env.update_reveal_probabilities();

        // 阵亡计数（按型）+ 池（同型聚集的规范序，仅供展示）
        env.dead_piece_counts_by_type = s.dead_counts;
        for player_idx in 0..2 {
            let mut idx = 0usize;
            for pt in 0..NUM_PIECE_TYPES_MAX {
                for _ in 0..s.dead_counts[player_idx][pt] {
                    if idx >= MAX_PIECES_PER_PLAYER {
                        return Err(EnvError::InvalidSnapshot { context: "快照阵亡计数超出池容量" });
                    }
                    env.dead_pieces_pool[player_idx][idx] = PieceType::from_index(pt);
                    idx += 1;
                }
            }
            env.dead_pieces_count[player_idx] = idx;
        }

        env.current_player = s.current_player;
        env.move_counter = s.move_counter as usize;
        env.total_step_counter = s.total_step_counter as usize;
        env.scores = s.scores;
        env.last_action = s.last_action;
        Ok(env)
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::env::GameEnv;
    use crate::core::env::seed::SeedableEnv;

    fn legal_actions(env: &DarkChessEnv) -> Vec<usize> {
        let mut masks = vec![0i32; env.action_space_size()];
        env.action_masks_into(&mut masks);
        masks
            .iter()
            .enumerate()
            .filter(|(_, m)| **m == 1)
            .map(|(i, _)| i)
            .collect()
    }

    fn features(env: &DarkChessEnv) -> (Vec<f32>, Vec<f32>) {
        let mut board = Vec::new();
        let mut scalars = Vec::new();
        env.encode_resnet_features_flat_into(&mut board, &mut scalars);
        (board, scalars)
    }

    /// 编码 → 解码 → 再编码：字节串必须逐位相同（编码器/解码器互逆）。
    fn assert_codec_roundtrip(env: &DarkChessEnv) {
        let bytes = env.to_snapshot().encode();
        let decoded = PositionSnapshot::decode(&bytes).expect("快照解码失败");
        assert_eq!(decoded, env.to_snapshot(), "快照解码结果与原状态不一致");
        assert_eq!(decoded.encode(), bytes, "重编码字节串不一致");
    }

    /// 所有变体：走若干步后快照应能精确重建静态状态（特征逐位相同）。
    #[test]
    fn snapshot_rebuilds_static_state_for_all_variants() {
        for variant in Variant::ALL {
            let mut env = DarkChessEnv::with_config(variant.config());
            for step in 0..10 {
                let legal = legal_actions(&env);
                if legal.is_empty() {
                    break;
                }
                let action = legal[step % legal.len()];
                if env.step(action, None).is_err() {
                    break;
                }
            }
            assert_codec_roundtrip(&env);

            let rebuilt = DarkChessEnv::from_snapshot(&env.to_snapshot()).expect("重建失败");
            assert_eq!(features(&env), features(&rebuilt), "{variant:?} 特征不一致");
            assert_eq!(
                legal_actions(&env),
                legal_actions(&rebuilt),
                "{variant:?} 动作掩码不一致"
            );
            assert_eq!(
                env.get_reveal_probabilities(),
                rebuilt.get_reveal_probabilities(),
                "{variant:?} 揭示概率表不一致"
            );
            assert_eq!(
                env.check_game_over_conditions(),
                rebuilt.check_game_over_conditions(),
                "{variant:?} 终局判定不一致"
            );
            for player in [Player::Red, Player::Black] {
                assert_eq!(env.get_hp(player), rebuilt.get_hp(player), "{variant:?} 血量不一致");
                assert_eq!(
                    env.get_dead_pieces(player).len(),
                    rebuilt.get_dead_pieces(player).len(),
                    "{variant:?} 阵亡计数不一致"
                );
            }
        }
    }

    /// 设种子对局：重建后与原环境**同步续走**，轨迹必须逐位一致。
    /// 这是最强的一致性检查 —— 任何漏存的状态字段都会在若干步内表现为特征/终局判定分叉。
    #[test]
    fn seeded_snapshot_replays_identically() {
        for variant in Variant::ALL {
            let mut env = DarkChessEnv::with_config(variant.config());
            env.set_seed(20260916);
            for step in 0..8 {
                let legal = legal_actions(&env);
                if legal.is_empty() {
                    break;
                }
                let action = legal[step % legal.len()];
                if env.step(action, None).is_err() {
                    break;
                }
            }

            let mut rebuilt =
                DarkChessEnv::from_snapshot(&env.to_snapshot()).expect("重建失败");

            for step in 0..60 {
                let legal = legal_actions(&env);
                if legal.is_empty() {
                    break;
                }
                let action = legal[step % legal.len()];
                let r1 = env.step(action, None);
                let r2 = rebuilt.step(action, None);
                assert_eq!(r1.is_ok(), r2.is_ok(), "{variant:?} 第 {step} 步可执行性分叉");
                assert_eq!(features(&env), features(&rebuilt), "{variant:?} 第 {step} 步特征分叉");
                assert_eq!(
                    env.check_game_over_conditions(),
                    rebuilt.check_game_over_conditions(),
                    "{variant:?} 第 {step} 步终局判定分叉"
                );
                assert_eq!(
                    env.to_snapshot().encode(),
                    rebuilt.to_snapshot().encode(),
                    "{variant:?} 第 {step} 步完整状态分叉"
                );
                if env.check_game_over_conditions().2.is_some() {
                    break;
                }
            }
        }
    }

    /// 非法载荷必须返回 Err（不 panic、不静默接受）。
    #[test]
    fn invalid_snapshot_payloads_are_rejected() {
        let env = DarkChessEnv::new();
        let good = env.to_snapshot().encode();

        // 版本不符
        let mut bad_version = good.clone();
        bad_version[0] = SNAPSHOT_VERSION + 1;
        assert!(PositionSnapshot::decode(&bad_version).is_err());

        // 截断
        assert!(PositionSnapshot::decode(&good[..good.len() - 1]).is_err());
        assert!(PositionSnapshot::decode(&[]).is_err());

        // 槽位编码非法（slot 字节最大 15）
        let mut bad_slot = good.clone();
        bad_slot[2] = 200;
        assert!(PositionSnapshot::decode(&bad_slot).is_err());

        // 变体与槽位数不符：把 4x2 快照（8 槽位）标成 4x8（32 槽位）
        let mut snap = DarkChessEnv::new_mini().to_snapshot();
        snap.variant = Variant::DarkChess4x8;
        assert!(DarkChessEnv::from_snapshot(&snap).is_err());

        // 阵亡计数超出池容量
        let mut snap = env.to_snapshot();
        snap.dead_counts[0][0] = u8::MAX;
        assert!(DarkChessEnv::from_snapshot(&snap).is_err());
    }
}
