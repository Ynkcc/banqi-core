// src/game_env/board/reset.rs
// 内部状态复位、棋盘初始化、翻子处理、翻棋概率表。

use super::*;
use crate::core::env::error::EnvError;

impl DarkChessEnv {
    pub(crate) fn reset_internal_state(&mut self) {
        let (config, seed, true_board) = (self.config, self.seed, self.true_board);
        *self = Self::fresh_with(config, [Slot::Empty; MAX_POSITIONS], seed, true_board);
    }

    /// 初始化棋盘布局 (Shuffle Bag Model)
    pub(crate) fn initialize_board(&mut self) {
        let cfg = self.config; // Copy，避免对 self 的长期借用
        let mut rng_std = self.seed.map(rand::rngs::StdRng::seed_from_u64);

        // 1. 生成实际棋子池 (写入 Buffer)：按激活类型、每类数量
        let mut idx = 0;
        for &player in &[Player::Red, Player::Black] {
            for &pt_idx in cfg.active_types.iter().take(cfg.num_active) {
                let piece_type = PieceType::from_index(pt_idx);
                for _ in 0..cfg.piece_counts[pt_idx] {
                    self.hidden_pieces_pool[idx] = Piece::new(piece_type, player);
                    idx += 1;
                }
            }
        }
        self.hidden_pieces_count = idx;

        // 打乱 slice
        if let Some(ref mut rng) = rng_std {
            self.hidden_pieces_pool[0..self.hidden_pieces_count].shuffle(rng);
            let mut tb = [Piece::default(); MAX_POSITIONS];
            let count = self.hidden_pieces_count;
            tb[..count].copy_from_slice(&self.hidden_pieces_pool[0..count]);
            self.true_board = Some(tb);
        } else {
            let mut rng = thread_rng();
            self.hidden_pieces_pool[0..self.hidden_pieces_count].shuffle(&mut rng);
            self.true_board = None;
        }

        // 2. 填充棋盘
        self.empty_bitboard = 0;
        let bmask = board_mask(&cfg);
        self.hidden_bitboard = bmask;

        for sq in 0..cfg.total_positions {
            self.board[sq] = Slot::Hidden;
        }

        self.update_reveal_probabilities();

        // 3. 随机翻开 N 个 Hidden 位置
        if cfg.total_positions > 0 {
            let mut hidden_indices: Vec<usize> = (0..cfg.total_positions).collect();
            if let Some(ref mut rng) = rng_std {
                hidden_indices.shuffle(rng);
            } else {
                let mut rng = thread_rng();
                hidden_indices.shuffle(&mut rng);
            }
            let reveal_count = std::cmp::min(hidden_indices.len(), cfg.initial_revealed_pieces);

            for &idx in hidden_indices.iter().take(reveal_count) {
                if let Err(e) = self.reveal_piece_at(idx, None) {
                    eprintln!("⚠️ initialize_board 预翻棋子失败 (sq={idx}): {e}");
                    break;
                }
            }
        }
    }

    /// 翻开指定位置的棋子并更新 Bitboards
    pub(crate) fn reveal_piece_at(
        &mut self,
        sq: usize,
        specified_piece: Option<Piece>,
    ) -> Result<(), EnvError> {
        if !matches!(self.board[sq], Slot::Hidden) {
            return Err(EnvError::BrokenInvariant {
                context: "尝试翻开非暗子位置",
            });
        }

        if self.hidden_pieces_count == 0 {
            return Err(EnvError::BrokenInvariant {
                context: "棋盘有暗子但隐藏棋子池已空",
            });
        }

        // 获取 slice 视图
        let active_slice = &self.hidden_pieces_pool[0..self.hidden_pieces_count];

        let idx = if let Some(target) = specified_piece {
            match active_slice.iter().position(|p| *p == target) {
                Some(idx) => idx,
                None => {
                    return Err(EnvError::BrokenInvariant {
                        context: "指定的棋子不在隐藏棋子池中",
                    });
                }
            }
        } else if let Some(tb) = self.true_board {
            let target = tb[sq];
            match active_slice.iter().position(|p| *p == target) {
                Some(idx) => idx,
                None => {
                    return Err(EnvError::BrokenInvariant {
                        context: "真实棋盘指定棋子不在隐藏池",
                    });
                }
            }
        } else {
            let mut rng = thread_rng();
            rng.gen_range(0..self.hidden_pieces_count)
        };

        // Swap Remove 逻辑 (Copy version)
        let last_idx = self.hidden_pieces_count - 1;
        self.hidden_pieces_pool.swap(idx, last_idx); // 将选中的棋子交换到末尾
        let piece = self.hidden_pieces_pool[last_idx]; // 取出
        self.hidden_pieces_count -= 1; // 缩小有效范围

        // 更新 Bitboards
        let mask = ull(sq);
        self.hidden_bitboard &= !mask;

        let p_bb = &mut self.revealed_bitboards[piece.player.idx()];
        *p_bb |= mask;

        let pt_bb = &mut self.piece_bitboards[piece.player.idx()][piece.piece_type as usize];
        *pt_bb |= mask;

        self.board[sq] = Slot::Revealed(piece);
        self.last_revealed_piece = Some(piece);
        self.update_reveal_probabilities();
        Ok(())
    }

    /// 由当前暗子袋重算揭示概率表（`snapshot::from_snapshot` 还原状态时复用）。
    pub(crate) fn update_reveal_probabilities(&mut self) {
        let cfg = &self.config;
        let total_hidden = self.hidden_pieces_count;

        if total_hidden == 0 {
            self.reveal_probabilities = [0.0; MAX_REVEAL_PROBABILITY_SIZE];
            return;
        }

        let mut counts = vec![0; cfg.reveal_probability_size];
        for i in 0..total_hidden {
            let piece = self.hidden_pieces_pool[i];
            let id = cfg.outcome_id_for(piece.piece_type, piece.player == Player::Black);
            counts[id] += 1;
        }

        for i in 0..cfg.reveal_probability_size {
            self.reveal_probabilities[i] = counts[i] as f32 / total_hidden as f32;
        }
    }

    pub fn get_reveal_probabilities(&self) -> &[f32] {
        &self.reveal_probabilities[0..self.config.reveal_probability_size]
    }
}
