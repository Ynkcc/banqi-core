//! Expectimax 强引擎单元测试。

use crate::core::env::DarkChessEnv;
use crate::core::expectimax::nnue::test_support::DummyNnue;
use std::sync::Arc;

use super::search::{SearchConfig, search};

#[test]
fn engine_returns_legal_action() {
    let mut env = DarkChessEnv::new();
    env.seed = Some(5);
    env.reset();
    let cfg = SearchConfig {
        node_budget: 50_000,
        max_depth: 6,
        nnue_evaluator: Some(Arc::new(DummyNnue)),
        ..Default::default()
    };
    let res = search(&env, &cfg).expect("应返回动作");
    let mut masks = vec![0i32; env.config.action_space_size];
    env.action_masks_into(&mut masks);
    assert_eq!(masks[res.action], 1, "引擎返回非法动作 {}", res.action);
    assert!(res.depth >= 1);
    assert!(res.nodes <= cfg.node_budget + 1000);
}
