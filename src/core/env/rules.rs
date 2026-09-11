use super::actions::{action_lookup_tables, pack_coords};
use super::bitboard::{board_mask, msb_index, not_file_a, not_file_h, pop_lsb, ray_attacks, trailing_zeros, ull};
use super::board::DarkChessEnv;
use super::config::NUM_PIECE_TYPES_MAX;
use super::types::*;

// ==============================================================================
// --- 规则逻辑扩展块 (动作掩码、胜负判定) ---
// ==============================================================================

impl DarkChessEnv {
    /// 游戏终止条件检查
    pub fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        let cfg = &self.config;
        let mut masks = vec![0i32; cfg.action_space_size];
        self.get_action_masks_for_player_into(self.get_current_player(), &mut masks);
        let moves_empty = masks.iter().all(|&x| x == 0);
        self.check_game_over_with_moves(moves_empty)
    }

    /// 终局判定核心：`moves_empty` 为「当前走子方是否无合法走法」。
    /// 调用方若已生成过走子列表（如搜索），直接传 `moves.is_empty()`，
    /// 避免重复计算动作掩码。返回 `(terminated, truncated, winner)`。
    pub fn check_game_over_with_moves(&self, moves_empty: bool) -> (bool, bool, Option<i32>) {
        let cfg = &self.config;
        if self.get_score(Player::Red) <= 0 {
            return (true, false, Some(Player::Black.val()));
        }
        if self.get_score(Player::Black) <= 0 {
            return (true, false, Some(Player::Red.val()));
        }

        // 全灭判定 (使用 count 判断)
        if self.get_dead_pieces(Player::Red).len() == cfg.total_pieces_per_player {
            return (true, false, Some(Player::Black.val()));
        }
        if self.get_dead_pieces(Player::Black).len() == cfg.total_pieces_per_player {
            return (true, false, Some(Player::Red.val()));
        }

        // 无合法走法
        if moves_empty {
            return (
                true,
                false,
                Some(self.get_current_player().opposite().val()),
            );
        }

        if self.get_move_counter() >= cfg.max_consecutive_moves_for_draw {
            return (true, false, Some(0));
        }

        if self.get_total_steps() >= cfg.max_steps_per_episode {
            return (false, true, Some(0));
        }

        (false, false, None)
    }

    // --- 动作掩码计算 ---

    pub fn action_masks(&self) -> Vec<i32> {
        let mut mask = vec![0; self.config.action_space_size];
        self.action_masks_into(&mut mask);
        mask
    }

    pub fn legal_action_indices(&self) -> Vec<usize> {
        let mut mask = vec![0; self.config.action_space_size];
        self.action_masks_into(&mut mask);
        (0..self.config.action_space_size)
            .filter(|&i| mask[i] == 1)
            .collect()
    }

    pub fn action_masks_into(&self, mask: &mut [i32]) {
        self.get_action_masks_for_player_into(self.get_current_player(), mask);
    }

    /// 结构化走法生成：在动作掩码基础上补充坐标与语义标记
    /// （翻棋 / 吃明子 / 机会动作），供搜索走法排序使用。
    pub fn generate_moves(&self, player: Player) -> Vec<Move> {
        let cfg = &self.config;
        let mut mask = vec![0i32; cfg.action_space_size];
        self.get_action_masks_for_player_into(player, &mut mask);
        let lookup = action_lookup_tables(cfg);
        let hidden_bb = self.get_hidden_bitboard();
        let opp_revealed_bb = self.get_revealed_bitboards()[player.opposite().idx()];

        (0..cfg.action_space_size)
            .filter(|&a| mask[a] == 1)
            .map(|action| {
                let coords = &lookup.action_to_coords[action];
                if coords.len() == 1 {
                    Move {
                        action,
                        from: coords[0],
                        to: coords[0],
                        is_chance: true,
                        is_capture: false,
                        is_flip: true,
                    }
                } else {
                    let (from, to) = (coords[0], coords[1]);
                    let is_chance = (hidden_bb & ull(to)) != 0;
                    Move {
                        action,
                        from,
                        to,
                        is_chance,
                        is_capture: !is_chance && (opp_revealed_bb & ull(to)) != 0,
                        is_flip: false,
                    }
                }
            })
            .collect()
    }

    pub(super) fn get_action_masks_for_player_into(&self, player: Player, mask: &mut [i32]) {
        let cfg = &self.config;
        for m in mask.iter_mut() {
            *m = 0;
        }
        let lookup = action_lookup_tables(cfg);

        // 1. 翻棋动作
        let mut temp_hidden = self.get_hidden_bitboard();
        while temp_hidden != 0 {
            let sq = pop_lsb(&mut temp_hidden);
            if let Some(&idx) = lookup.coords_to_action.get(&pack_coords(&[sq])) {
                mask[idx] = 1;
            }
        }

        let empty_bb = self.get_empty_bitboard();
        let my = player;
        let opp = player.opposite();

        let my_revealed_bb = self.get_revealed_bitboards()[my.idx()];
        let my_piece_bb = self.get_piece_bitboards()[my.idx()];
        let opp_piece_bb = self.get_piece_bitboards()[opp.idx()];

        // 2. 常规移动（仅遍历激活的棋子类型；未激活类型位棋盘为空，自然跳过）
        let mut target_bbs: [u64; NUM_PIECE_TYPES_MAX] = [0; NUM_PIECE_TYPES_MAX];
        let mut cumulative_targets: u64 = empty_bb;

        for &pt in cfg.active_types.iter().take(cfg.num_active) {
            cumulative_targets |= opp_piece_bb[pt];
            target_bbs[pt] = cumulative_targets;
        }

        // 兵克将 / 将怕兵 特例（类型索引固定：兵=0，将=6）
        let soldier = PieceType::Soldier as usize;
        let general = PieceType::General as usize;
        target_bbs[soldier] |= opp_piece_bb[general];
        target_bbs[general] &= !opp_piece_bb[soldier];

        let bmask = board_mask(cfg);
        let nfa = not_file_a(cfg);
        let nfh = not_file_h(cfg);
        let shifts = [-(cfg.cols as isize) as i32, (cfg.cols as i32), -1, 1];
        let wrap_checks = [bmask, bmask, nfa, nfh];

        for &pt in cfg.active_types.iter().take(cfg.num_active) {
            if pt == PieceType::Cannon as usize {
                continue;
            }

            let from_bb = my_piece_bb[pt];
            if from_bb == 0 {
                continue;
            }

            for (dir_idx, &shift) in shifts.iter().enumerate() {
                let wrap = wrap_checks[dir_idx];

                let temp_from_bb = from_bb & wrap;
                if temp_from_bb == 0 {
                    continue;
                }

                let potential_to_bb = if shift > 0 {
                    (temp_from_bb << (shift as u32)) & bmask
                } else {
                    (temp_from_bb >> ((-shift) as u32)) & bmask
                };

                let mut actual_to_bb = potential_to_bb & target_bbs[pt];

                while actual_to_bb != 0 {
                    let to_sq = pop_lsb(&mut actual_to_bb);

                    let from_sq = if shift > 0 {
                        (to_sq as isize - (shift as isize)) as usize
                    } else {
                        (to_sq as isize + ((-shift) as isize)) as usize
                    };

                    if let Some(&idx) =
                        lookup.coords_to_action.get(&pack_coords(&[from_sq, to_sq]))
                    {
                        mask[idx] = 1;
                    }
                }
            }
        }

        // 3. 炮击
        let my_cannons_bb = my_piece_bb[PieceType::Cannon as usize];
        if my_cannons_bb != 0 {
            let all_pieces_bb = bmask & !empty_bb;

            let valid_cannon_targets = bmask & (!my_revealed_bb);

            let mut temp_cannons = my_cannons_bb;
            let rays = ray_attacks(cfg);
            while temp_cannons != 0 {
                let from_sq = pop_lsb(&mut temp_cannons);

                for dir in 0..4 {
                    let ray_bb = rays[dir][from_sq];
                    let blockers = ray_bb & all_pieces_bb;

                    if blockers == 0 {
                        continue;
                    }

                    let Some(screen_sq) = first_blocker(dir, blockers) else {
                        continue;
                    };

                    let after_screen_ray = rays[dir][screen_sq];
                    let targets = after_screen_ray & all_pieces_bb;

                    if targets == 0 {
                        continue;
                    }

                    let Some(target_sq) = first_blocker(dir, targets) else {
                        continue;
                    };

                    if ((ull(target_sq)) & valid_cannon_targets) != 0 {
                        if let Some(&idx) =
                            lookup.coords_to_action.get(&pack_coords(&[from_sq, target_sq]))
                        {
                            mask[idx] = 1;
                        }
                    }
                }
            }
        }
    }
}

/// 炮击路径上距离 from 最近的一个阻挡子：
/// UP/LEFT（dir 0|2）沿高位方向取 msb，DOWN/RIGHT 取 lsb。
fn first_blocker(dir: usize, bb: u64) -> Option<usize> {
    match dir {
        0 | 2 => msb_index(bb),
        _ => Some(trailing_zeros(bb)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    /// `generate_moves` 与 `action_masks_into` 的动作集合逐位一致。
    fn assert_moves_match(env: &DarkChessEnv) {
        let mut masks = vec![0i32; env.config.action_space_size];
        env.action_masks_into(&mut masks);
        let mut gen_actions = vec![0i32; env.config.action_space_size];
        for m in env.generate_moves(env.get_current_player()) {
            gen_actions[m.action] = 1;
        }
        assert_eq!(
            masks, gen_actions,
            "generate_moves 与 action_masks 不一致 (player={})",
            env.get_current_player()
        );
    }

    /// 多 seed 随机对局逐步校验生成动作与掩码始终一致。
    #[test]
    fn generate_moves_matches_action_masks() {
        for seed in 1..=12u64 {
            let mut env = DarkChessEnv::new();
            env.seed = Some(seed);
            env.reset();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
            for step in 0..80 {
                assert_moves_match(&env);
                let mut masks = vec![0i32; env.config.action_space_size];
                env.action_masks_into(&mut masks);
                let legal: Vec<usize> = (0..env.config.action_space_size)
                    .filter(|&i| masks[i] == 1)
                    .collect();
                if legal.is_empty() {
                    break;
                }
                let action = legal[rng.gen_range(0..legal.len())];
                match env.step(action, None) {
                    Ok((_, terminated, _, _)) => {
                        if terminated {
                            break;
                        }
                    }
                    Err(e) => panic!("seed={seed} step={step}: {e}"),
                }
            }
        }
    }

    /// 元属性抽查：`Move` 的语义标记与 `is_chance_action` 一致，动作可执行。
    #[test]
    fn move_metadata_is_consistent() {
        let mut env = DarkChessEnv::new();
        env.seed = Some(7);
        env.reset();
        for m in env.generate_moves(env.get_current_player()) {
            assert_eq!(env.is_chance_action(m.action), m.is_chance, "action={}", m.action);
            if m.is_flip {
                assert!(m.is_chance && m.from == m.to);
            }
            if m.is_capture {
                assert!(!m.is_chance);
            }
            let mut next = env;
            assert!(next.step(m.action, None).is_ok(), "非法动作 {}", m.action);
        }
    }
}
