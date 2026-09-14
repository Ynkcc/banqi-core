// src/mcts/search.rs
// Gumbel AlphaZero MCTS 搜索器定义、构造与树推进（泛型化：G = 游戏环境）
//
// 分层说明：
// - 本文件保留「搜索器结构体定义 + 构造 + 树推进（step_next）」；
// - 根准备 / 结果组装见 root.rs，路径选择见 path_select.rs，
//   主搜索循环见 run.rs，树构建与价值回溯见 tree.rs。

use crate::core::env::GameEnv;

use super::config::GumbelConfig;
use super::evaluator::Evaluator;
use super::node::{MctsArena, MctsNode};
use rand::prelude::*;

/// Gumbel MCTS 搜索器
///
/// 管理 MCTS 树的构建、搜索和动作选择过程。
/// 泛型 `G` 为游戏环境（须实现 `GameEnv`），泛型 `E` 必须实现 `Evaluator<G>`。
pub struct GumbelMCTS<'a, G: GameEnv, E: Evaluator<G>> {
    /// Arena 内存池
    pub arena: MctsArena<G>,
    /// 搜索树的根节点在 Arena 中的索引
    pub root_idx: usize,
    /// 状态评估器
    pub(crate) evaluator: &'a E,
    /// 搜索配置
    pub(crate) config: GumbelConfig,
    /// Scratch pad: 用于 Gumbel 采样阶段的临时存储，避免反复堆分配
    /// Vec<(action_index, gumbel_noise_logit)>
    pub(crate) scratch_gumbel: Vec<(usize, f32)>,
    /// 根节点合法动作掩码：在 run() 入口处计算一次，搜索全程不可变。
    /// 作为根节点合法动作的权威来源，直接用于结果返回。
    pub(crate) root_action_mask: Vec<i32>,
    /// 遍历临时缓冲：select_path_collect 中沿路径向下时复用，
    /// 存储当前遍历节点的 action mask。与 root_action_mask 物理隔离。
    pub(crate) traversal_action_mask: Vec<i32>,
    /// 动作空间大小：由初始环境的 `config` 决定（4x8 / 4x4 / 4x2 各不相同），
    /// 搜索期间环境不变，故在构造时固定。
    pub(crate) action_space: usize,
    /// 复用的随机数生成器，避免每次搜索/采样重建 thread_rng
    pub(crate) rng: StdRng,
}

impl<'a, G: GameEnv, E: Evaluator<G>> GumbelMCTS<'a, G, E> {
    /// 创建一个新的 GumbelMCTS 实例
    ///
    /// 初始化根节点并准备搜索。
    pub fn new(env: &G, evaluator: &'a E, config: GumbelConfig) -> Self {
        let mut arena = MctsArena::new();
        let state = env.get_resnet_state();
        let root_node = MctsNode::new(1.0, 0.0, false, Some(*env), Some(state), true);
        let root_idx = arena.allocate(root_node);
        let action_space = env.action_space_size();

        Self {
            arena,
            root_idx,
            evaluator,
            config,
            scratch_gumbel: Vec::with_capacity(32),
            root_action_mask: vec![0; action_space],
            traversal_action_mask: vec![0; action_space],
            action_space,
            rng: StdRng::from_entropy(),
        }
    }

    /// 当前根节点持有的环境引用（终局血量差等终局信息使用）。
    pub fn root_env(&self) -> Option<&G> {
        self.arena.get(self.root_idx).env.as_ref()
    }

    /// 当前根节点环境的值拷贝（根节点恒持有环境，缺失时不 panic）。
    pub(crate) fn root_env_copied(&self) -> Option<G> {
        self.arena.get(self.root_idx).env
    }

    /// 将搜索树移动到下一个状态
    ///
    /// 当环境发生实际变动（例如玩家采取了某个动作）时调用。
    /// 该方法会尝试重用现有的子树，如果子节点不存在则创建新的根节点。
    pub fn step_next(&mut self, env: &G, action: usize) {
        let root_node = self.arena.get(self.root_idx);

        // 查找子节点
        let child_idx = root_node.child_idx(action);

        if let Some(idx) = child_idx {
            let child = self.arena.get(idx);
            if child.is_chance_node {
                // 如果是机会节点 (翻牌)，需要根据实际翻出的棋子选择对应的子节点
                if let Some(outcome_id) = env.step_outcome_id(action) {
                    if let Some(next_idx) = child.outcome_idx(outcome_id) {
                        self.root_idx = next_idx;
                        let next_node = self.arena.get_mut(next_idx);
                        next_node.is_root_node = true;
                        return;
                    }
                }
            } else {
                // 普通节点，直接移动根节点
                self.root_idx = idx;
                let next_node = self.arena.get_mut(idx);
                next_node.is_root_node = true;
                return;
            }
        }

        // 如果无法重用子树，则重置根节点
        let state = env.get_resnet_state();
        let mut new_root = MctsNode::new(1.0, 0.0, false, Some(*env), Some(state), true);
        new_root.is_root_node = true;
        self.root_idx = self.arena.allocate(new_root);
    }
}
