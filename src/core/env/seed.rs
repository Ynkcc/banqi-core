// 环境种子设置与底层棋盘访问辅助 trait（自 pipeline/self_play/match_core.rs 迁入，
// 因需访问 DarkChessEnv 的 pub(crate) 内部，必须与环境同 crate）。

use crate::core::env::{DarkChessEnv, Game4x4Env, MiniDarkChessEnv};

pub trait AsDarkChessRef {
    fn as_darkchess_ref(&self) -> &DarkChessEnv;
}

impl AsDarkChessRef for DarkChessEnv {
    fn as_darkchess_ref(&self) -> &DarkChessEnv {
        self
    }
}

// 暗棋变体均为 DarkChessEnv 的包装，统一通过 inner 委托。
macro_rules! impl_as_darkchess_ref_for_variant {
    ($ty:ty) => {
        impl AsDarkChessRef for $ty {
            fn as_darkchess_ref(&self) -> &DarkChessEnv {
                &self.inner
            }
        }
    };
}

impl_as_darkchess_ref_for_variant!(MiniDarkChessEnv);
impl_as_darkchess_ref_for_variant!(Game4x4Env);

pub trait SeedableEnv {
    fn set_seed(&mut self, seed: u64);
}

impl SeedableEnv for DarkChessEnv {
    fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
        self.reset_internal_state();
        self.initialize_board();
    }
}

// 变体包装统一通过 AsDarkChessRef + &mut 委托给内部 DarkChessEnv。
// （变体与 DarkChessEnv 字段同构，inner 可安全可变借用。）
macro_rules! impl_seedable_for_variant {
    ($ty:ty) => {
        impl SeedableEnv for $ty {
            fn set_seed(&mut self, seed: u64) {
                // 通过字段重投影获得 &mut DarkChessEnv（变体仅包装 inner）。
                let inner = &mut self.inner;
                SeedableEnv::set_seed(inner, seed);
            }
        }
    };
}

impl_seedable_for_variant!(MiniDarkChessEnv);
impl_seedable_for_variant!(Game4x4Env);
