# Banqi Core 架构文档

> **维护约定**：本文档是代码库的结构性快照，供 AI 助手与人类快速定位，避免每次需求变更都重新全库探索。
>
> **更新规则**：
> - 阅读实际代码发现与本文档描述不一致时，更新本文档；
> - 仅函数级内部改动不需更新（本文只记录"结构性事实"：模块划分、关键类型、入口、数据流）；
> - **触发更新的结构性变更**（任一发生即须更新本文档）：
>   1. 新增 / 删除 / 重命名模块、源文件或子目录；
>   2. 新增 / 删除公开类型、trait、常量、feature 标志位、公开函数入口；
>   3. 模块间依赖方向变化（如 A 不再依赖 B、新增跨模块引用）；
>   4. 端到端数据流变化（如新增一次网络评估通道、改变机会节点语义、改变动作选择流程）；
>   5. 新增 / 删除游戏变体、`GameConfig` 字段语义变化、动作空间组成变化；
>   6. 新增 bin / 子包 / 跨语言（PyO3、Tauri、gRPC）桥接接口，或改动 trait 契约边界。
> - **不触发更新的非结构性变更**：私有函数增删改、算法参数默认值调整、注释与日志文案、测试用例增删、缓存 / 排序键等内部实现细节。
> - **更新粒度与写法**：只描述"是什么、在哪、如何连接"，不复述实现步骤；关键类型用列表 / 表格，数据流用有序步骤；命名严格与代码一致（模块路径、类型名、函数名）。
> - **同步要求**：文档为仓库根 `ARCHITECTURE.md`；结构性变更应在同一提交内更新，并在文末"变更记录"追加一行（日期 + 摘要 + 影响范围）。
> - **事实优先级**：代码是唯一事实来源。文档与代码冲突时以代码为准，并立即修正文档。
> - **禁止事项**：不要把临时调试、TODO、未落地设计写进本文档；不要复制大段源码。

---

## 变更记录

| 日期 | 变更摘要 | 影响范围 |
|------|----------|----------|
| 2026-09-11 | 首次成文，基于当前 `src/core` 快照梳理模块划分、关键类型、入口与数据流 | 全库 |
| 2026-09-11 | `DarkChessEnv` 新增只读访问器 `last_revealed_piece()`（NNUE 增量差分等上层 crate 消费） | `env/board/struct_def.rs` |
| 2026-09-11 | 新增 `env::variants::CurriculumEnv` trait（`with_initial_revealed(n)`，覆盖初始预翻棋子数；实现于 DarkChessEnv / Game4x4Env / MiniDarkChessEnv，棋盘/动作空间/特征维度不变，服务课程学习） | `env/variants/mod.rs` |
| 2026-09-13 | `GameEnv::action_space_size()` 由关联函数改为 `&self` 方法（返回 `config.action_space_size`）：`DarkChessEnv` 同类型可对应 4x8 / 4x4 / 4x2，动作空间须随变体变化；`GumbelMCTS` 新增 `action_space` 字段（构造时冻结）替代各处的 `G::action_space_size()`，掩码/策略/logits 缓冲随变体收敛（4x2 由 352 降到 40），变体包装固有方法同步改为 `&self` | `env/traits.rs`、`env/variants/mod.rs`、`env/variants/tic_tac_toe.rs`、`mcts/search.rs`、`mcts/tree.rs`、`mcts/policy.rs` |
| 2026-09-13 | 修复 `get_resnet_state` 变体 panic 并移除虚假形状常量：①`GameEnv` 删除 `RESNET_BOARD_CHANNELS` / `BOARD_ROWS` / `BOARD_COLS` / `RESNET_SCALAR_FEATURE_COUNT` 四个关联常量（`DarkChessEnv` 可运行时切换 4x8 / 4x4 / 4x2，编译期常量无法表达，曾导致非 4x8 变体编码长度与常量不匹配而重塑 panic）；②`get_resnet_state` 改为必需方法：`DarkChessEnv` 由 `config` 推导形状，变体包装转发 inner，`TicTacToeEnv` 用自身常量；③新增 `ResNetObservation::from_flat` 统一重塑与报错信息；④新增回归测试 `resnet_state_shape_follows_config` | `env/traits.rs`、`env/features.rs`、`env/types.rs`、`env/variants/mod.rs`、`env/variants/tic_tac_toe.rs`、`env/board/tests.rs` |
| 2026-09-11 | 去重重构：①新增 `env/cache.rs` 泛型 `OnceLock` 缓存（`global_cache!` / `cached`），actions/bitboard/symmetry 四套缓存样板收敛；②变体包装（Game4x4Env / MiniDarkChessEnv）的常量、固有方法与 `GameEnv` 委托实现由 `variants::impl_darkchess_variant!` 宏统一生成，变体常量改由 const fn 配置函数推导；③`GameEnv::get_resnet_state` 提供基于 `encode_resnet_features_flat_into` 的默认实现，删除 DarkChessEnv / TicTacToeEnv 重复 impl；④`config.rs` 提取共享构建器 `make_config`，三变体配置收敛；⑤MCTS run 与 batched 双路径共享根准备（`refresh_root_action_mask` / `root_logits` / `sample_root_candidates`）、根扩展（`apply_root_eval`）、叶回填（`apply_leaf_evals`）与结果组装（`build_result`），`MctsNode` 新增 `child_idx` / `outcome_idx` 查找方法；⑥expectimax 提取 `require_nnue` / `feature_dim_ok` / `move_to_front` / `iterative_deepen`；⑦`PieceType::value()` 删除，走子排序统一使用 `GameConfig.piece_values`（修复 4x4 变体下排序分值与扣血不一致） | 全库（env / mcts / expectimax） |
| 2026-09-13 | 变体常量去重：删除 `env/constants.rs`（4x8 硬编码重复常量，仅方向常量移入 `bitboard.rs`）；`GameEnv::max_steps` 由关联函数改为 `&self` 方法、由 `config.max_steps_per_episode` 驱动（`banqi-engine` 策略 / `banqi-collector` 自对弈同步改用 `env.config`，不再依赖 4x8 编译期常量） | `env/mod.rs`、`env/bitboard.rs`、`env/traits.rs`、`env/variants/*` |
| 2026-09-14 | `DarkChessEnv` 新增只读访问器 `get_last_action()`（产生当前局面的最后一步动作，初始局面返回 `None`；供 GUI 还原 MCTS 树节点的到达着法） | `env/board/accessors.rs` |
| 2026-09-14 | 代码质量重构（一）：①新增 `env/error.rs::EnvError`，`GameEnv::step` 与 `DarkChessEnv::step` 错误类型由 `String` 改为 `EnvError`（`IllegalAction` / `BrokenInvariant`），`reveal_piece_at` 改为返回 `Result`，环境层热路径消除 `panic!`/`expect`；②MCTS 拆分 `search.rs`（681 行）为 `search.rs`（结构体 + `step_next`）/ `root.rs` / `path_select.rs` / `run.rs`，并将 `select_path_collect`（203 行）与 `run`（138 行）拆为多个辅助方法；③`tree.rs` 的 `get_node_idx_by_path` / `backprop_from_path` 改为返回 `Option`，消除路径缺失时的 panic；④`sampling.rs` 的 `sample_outcome_id` 改为 `sample_outcome`（返回 `(outcome_id, 子节点索引)`，单次查找）；⑤expectimax 拆分 `search.rs`（594 行）为 `search.rs`（入口 + Lazy SMP）/ `negamax.rs` / `iterative.rs` / `eval.rs` / `config.rs`；⑥新增 `Variant` 枚举（变体↔字符串↔棋盘尺寸↔配置单一真源），`GameConfig` 新增 `variant` 字段、`make_config` 改由变体推导行列；⑦新增契约测试 `variant_single_source_of_truth` / `illegal_action_returns_structured_error` | 全库（env / mcts / expectimax）；破坏性：`GameEnv::step` 错误类型变更，engine / collector / gui 已同步 |
| 2026-09-14 | `Variant` 变体成员定名 `DarkChess4x2`（原 `MiniDarkChess4x2`），字符串标识固定为 `"4x8"` / `"4x4"` / `"4x2"` 并作为 GUI 与前端共享的唯一变体词表（GUI 不再使用 `"dark"` / `"mini"`） | `env/config.rs`；下游 `banqi-gui`（Rust + `frontend/src`）已同步 |
| 2026-09-15 | MCTS 评估契约改为可失败：新增 `mcts::evaluator::EvaluatorError`（实现 `Display` + `Error`）；`Evaluator::evaluate` / `evaluate_logits` 由 `-> EvaluatorOutput` 改为 `-> Result<EvaluatorOutput, EvaluatorError>`；`GumbelMCTS::expand_root` 与 `GumbelMCTS::run` 同步改为 `run() -> Result<Option<MctsSearchResult>, EvaluatorError>`（`Ok(None)` = 无合法动作，`Err` = 评估失败）。目的：推理失败必须显式向上传播，禁止 panic 或静默退化为均匀策略 | `mcts/evaluator.rs`、`mcts/root.rs`、`mcts/run.rs`、`mcts/mod.rs`、`mcts/search_tests.rs`；破坏性：下游 `banqi-engine` / `banqi-collector` / `banqi-gui` 已同步 |
| 2026-09-15 | 热路径预计算表缓存去锁：`env/cache.rs` 新增 `VariantCache<T>`（按变体索引的 `OnceLock` 数组，`Variant` 新增 `index()`），`action_lookup_tables` / `ray_attacks` 的返回类型由 `Arc<T>` 改为 `&'static T`，读路径不再有 `Mutex::lock`（RMW）、哈希查找与 `Arc::clone`（RMW）；`global_cache!` / `cached` 保留给冷路径（symmetry 的 D4 置换表）。动机：4x2 自对弈 12 线程下采样显示 `action_lookup_tables` 16.5% + `Mutex::lock_contended` 14.7% + `ray_attacks` 4.4% + 哈希 1.4% ≈ 37% CPU 耗在全局锁缓存上，环境/搜索层扩展性被压到 ~1.4×；改后同一负载 155 → 677 局/s（4.2×） | `env/cache.rs`、`env/actions.rs`、`env/bitboard.rs`、`env/config.rs`、`env/symmetry.rs` |

---

## 1. 项目定位

`banqi-core`（crate 名 `banqi-core`）是暗棋（Dark Chess）的**领域核心库**，无二进制目标，仅 `src/lib.rs` 导出 `core` 模块。

- 职责：游戏规则（环境 / 走子生成 / 特征编码）、通用 Gumbel AlphaZero MCTS、Expectimax 强搜索引擎（含 NNUE 叶评估契约）、Zobrist 哈希。
- 边界：**神经网络后端**（TorchScript / ONNX）与**策略引擎 / UI / 训练管线**位于上层 crate（如 `banqi_4x8`、`banqi-engine`、`banqi-tauri`）与 Python 侧，通过本 crate 的 trait 对接：
  - `core::mcts::Evaluator`：MCTS 的批量策略 / 价值评估接口；
  - `core::expectimax::nnue::NnueEvaluate` / `NnueAccumulator`：Expectimax 的叶评估接口。
- 依赖（`Cargo.toml`）：`rand`、`rand_distr`、`ndarray`（serde）、`serde`、`slab`。Edition 2024。

```1:9:src/lib.rs
//! 神经网络后端（TorchScript / ONNX）与策略引擎在上层 crate `banqi_4x8` 中，
//! 通过本 crate 的 `core::mcts::Evaluator` trait 对接。
pub mod core;
```

---

## 2. 模块总览

```
src/
  lib.rs                      # crate 根：pub mod core
  core/
    mod.rs                    # 领域核心聚合：pub mod env / mcts / expectimax
    env/                      # 游戏环境（规则 / 状态 / 走子 / 特征 / 变体）
      mod.rs                  # 声明 + 公共 re-export
      types.rs                # PieceType / Player / Piece / Slot / Move / ResNetObservation
      config.rs               # Variant + GameConfig + 各变体配置（const fn + 共享构建器 make_config）+ 动作计数 + NNUE 布局推导
      error.rs                # EnvError（IllegalAction / BrokenInvariant）
      cache.rs                # 泛型 OnceLock 缓存：global_cache! 宏 + cached() 查/建/插
      traits.rs               # GameEnv trait + DarkChessEnv 实现 + 机会节点扩展点
      rules.rs                # 终局判定 + 动作掩码生成 + 结构化走子生成 + first_blocker
      features.rs             # StateView 投影 + ResNet 稠密特征 + NNUE 稀疏特征
      actions.rs              # 动作查找表（config 驱动 + 全局缓存）
      bitboard.rs             # 方向常量 + 位棋盘辅助 + 射线表（config 驱动 + 缓存）
      symmetry.rs             # D4 / Klein-4 空间对称（格子重排 / 动作置换）
      seed.rs                 # AsDarkChessRef / SeedableEnv trait
      board/
        mod.rs                # DarkChessEnv 聚合（reset 入口）
        struct_def.rs         # 结构体定义 + 构造器
        reset.rs              # 状态复位 / 棋盘初始化（袋模型）/ 翻子 / 概率表
        step.rs               # step 主逻辑 / 走子应用 / 机会节点扩展 / 打印
        accessors.rs          # 公共访问器 + pub(crate) 内部辅助
        tests.rs              # 单元测试
      variants/
        mod.rs                # 变体 re-export + impl_darkchess_variant! 统一实现宏 + CurriculumEnv
        game4x4.rs            # 4x4 暗棋（7 类全激活，每方 8 子；仅结构体 + 宏调用）
        mini_darkchess.rs     # 4x2 迷你暗棋（兵/炮/士/将，每方 4 子）
        tic_tac_toe.rs        # 井字棋（MCTS 泛型验证用，无机会节点）
    mcts/                     # Gumbel AlphaZero MCTS（泛型：G: GameEnv, E: Evaluator<G>）
      mod.rs                  # 声明 + re-export + 模块级注意事项
      search.rs               # GumbelMCTS 结构体定义 + 构造 + 树推进 step_next
      root.rs                 # 根准备（expand_root / 掩码 / 候选采样）+ 叶子回填 + 结果组装
      path_select.rs          # 路径选择 select_path_collect（机会/终局/步数兜底/PUCT）
      run.rs                  # 主搜索循环 run（Sequential Halving 编排与淘汰）
      config.rs               # GumbelConfig / MctsSearchResult
      evaluator.rs            # Evaluator trait / EvaluatorOutput / health_logits_expectation
      node.rs                 # MctsArena(Slab) / MctsNode / value_from_perspective
      path.rs                 # PathStep / PendingEval / SelectPathOutcome / ChanceSeed
      sampling.rs             # Gumbel Top-K 采样 / 机会结果采样
      tree.rs                 # 子节点构建 / 机会节点展开 / 价值回溯
      policy.rs               # probs-from-logits / 改进策略
      budget.rs               # SequentialHalvingBudget
      batched.rs              # BatchedTree：多树 lockstep 批量自对弈
      search_tests.rs         # 单元测试
    expectimax/               # Expecti-Alpha-Beta 强搜索（绑定 DarkChessEnv）
      mod.rs                  # ExpectimaxEngine + Ctx + FEAT_* 标志
      search.rs               # 搜索入口 search / search_par / Lazy SMP（+ SearchConfig 等再导出）
      negamax.rs              # 递归主体 negamax / quiesce / Star1 机会节点 / TT 探测与存储
      iterative.rs            # 迭代加深（根层单层搜索 + 逐层加深）
      eval.rs                 # 叶评估 eval_state / eval_acc + 增量累加器辅助
      config.rs               # SearchConfig / SearchResult（经 search 再导出）
      nnue.rs                 # NnueEvaluate / NnueAccumulator trait 契约（+ 测试 Dummy）
      ordering.rs             # 走子排序（MVV-LVA + 杀手 + 历史）+ 终局辅助
      zobrist.rs              # zkey / sym_zkey / TtEntry / 值域常量
      smp.rs                  # SharedTT：无锁共享置换表（Lazy SMP）
      tests.rs                # 单元测试
```

`core/mod.rs`：

```6:8:src/core/mod.rs
pub mod env;
pub mod expectimax;
pub mod mcts;
```

---

## 3. 关键类型速查

### 3.1 环境层（`core::env`）

| 类型 / 常量 | 位置 | 说明 |
|-------------|------|------|
| `PieceType` | `env::types` | 兵/炮/马/车/象/士/将（索引 0..6） |
| `Player` | `env::types` | `Red = 1` / `Black = -1`；`idx()` 0/1，`val()` 用于胜负比较 |
| `Piece` | `env::types` | `{ piece_type, player }`，`Copy` |
| `Slot` | `env::types` | `Empty` / `Hidden` / `Revealed(Piece)` |
| `Move` | `env::types` | 结构化走法：`action / from / to / is_chance / is_capture / is_flip` |
| `ResNetObservation` | `env::types` | `board: Array3<f32>` (C,H,W) + `scalars: Array1<f32>` |
| `GameConfig` | `env::config` | `Copy` 纯数据；决定变体 / 棋盘尺寸 / 子力 / 血量 / 动作空间 / 特征维度（含 `variant: Variant`） |
| `Variant` | `env::config` | 暗棋变体标识单一真源：`as_str()`（"4x8"/"4x4"/"4x2"）/ `from_str()` / `board_dims()` / `config()` |
| `EnvError` | `env::error` | `step` 错误类型：`IllegalAction` / `BrokenInvariant`（实现 `Display` + `std::error::Error`） |
| `darkchess_config` / `game_4x4_config` / `mini_config` | `env::config` | 三种暗棋变体配置 |
| `compute_action_counts` | `env::config` | 由 `(rows, cols)` 推导 `(reveal, regular, cannon)` 计数 |
| `DarkChessEnv` | `env::board` | `Copy` 环境，config 驱动；所有数组按 `MAX_*` 上界分配 |
| `GameEnv` | `env::traits` | MCTS 依赖的最小泛型接口 |
| `CurriculumEnv` | `env::variants` | 课程学习构造 trait：`with_initial_revealed(n)` 以指定初始翻子数创建变体环境（DarkChessEnv / Game4x4Env / MiniDarkChessEnv 均实现） |
| `Game4x4Env` / `MiniDarkChessEnv` | `env::variants` | 包装 `DarkChessEnv` + 变体关联常量 |
| `TicTacToeEnv` | `env::variants::tic_tac_toe` | 独立 `Copy` 环境，无机会节点，用于 MCTS 泛型验证 |
| `StateView` | `env::features` | `pub(crate)` 架构无关只读快照，ResNet 与 NNUE 特征的统一投影源 |
| `Symmetry` | `env::symmetry` | D4 八变换；`search_group` 按方盘/长方盘返回可用集 |

### 3.2 MCTS 层（`core::mcts`）

| 类型 | 位置 | 说明 |
|------|------|------|
| `GumbelMCTS<'a, G, E>` | `mcts::search` | 搜索器；持有 `MctsArena`、根索引、evaluator 借用、复用 rng / 掩码缓冲 |
| `MctsArena<G>` / `MctsNode<G>` | `mcts::node` | 基于 `Slab` 的节点池；节点保存 `env`（值语义快照）、`children`、`possible_states`（机会节点）、`prior/logit`、`value_sum/health_sum` |
| `GumbelConfig` | `mcts::config` | `num_simulations` / `max_considered_actions` / `c_scale` / `gumbel_scale` / 血量复合效用开关 |
| `MctsSearchResult` | `mcts::config` | 动作、观测、`improved_policy`、`mcts_value`、`completed_q`、访问数、掩码 |
| `Evaluator<G>` / `EvaluatorOutput` | `mcts::evaluator` | 批量评估契约：`evaluate -> Result<EvaluatorOutput, EvaluatorError>`（推理失败必须显式返回错误，不得 panic 或静默退化）；`health` 为可选血量分桶 logits |
| `EvaluatorError` | `mcts::evaluator` | 批量评估失败（后端报错 / 输出不符契约），实现 `Display` + `Error` |
| `PathStep` / `PendingEval<G>` / `ChanceSeed` | `mcts::path` | 路径步骤、待评估叶子、机会节点加权回传标记 |
| `SequentialHalvingBudget` | `mcts::budget` | 分阶段预算排程与淘汰计数 |
| `BatchedTree<'a, G, E>` | `mcts::batched` | 单棵批量自对弈树，`Root/Searching/Ready/Idle` 状态机 |

### 3.3 Expectimax 层（`core::expectimax`）

| 类型 / 常量 | 位置 | 说明 |
|-------------|------|------|
| `ExpectimaxEngine` | `expectimax` | 引擎实体，包裹 `SearchConfig`；`search` / `search_par` / `best_action` |
| `Ctx` | `expectimax` | 搜索上下文：节点计数、预算、杀手 / 历史、共享 TT、路径 zkey |
| `FEAT_ORDERING` / `FEAT_TT` / `FEAT_LMR` / `FEAT_REP` | `expectimax` | 特性位掩码 |
| `SearchConfig` / `SearchResult` | `expectimax::search` | 搜索配置（含 `nnue_evaluator`）/ 结果（动作、值、深度、节点数） |
| `NnueEvaluate` / `NnueAccumulator` | `expectimax::nnue` | 叶评估与增量累加器契约（实现位于上层 crate） |
| `TtEntry` / `zkey` / `sym_zkey` / `VMIN/VMAX/INF` | `expectimax::zobrist` | 置换表项、局面哈希、对称键、值域 |
| `SharedTT` | `expectimax::smp` | 原子打包的无锁共享置换表（Lazy SMP） |

---

## 4. 环境层（`core::env`）详解

### 4.1 设计主线：config 驱动 + Copy 值语义

- `GameConfig` 是 `Copy` 纯数据，随环境携带，决定"活跃范围"。
- `DarkChessEnv` 内部**所有定长数组按最大上界分配**（`MAX_POSITIONS = 32`、`NUM_PIECE_TYPES_MAX = 7`、`MAX_PIECES_PER_PLAYER = 16`、`MAX_REVEAL_PROBABILITY_SIZE = 14`），实际使用部分由 `config` 裁剪 —— 这是环境保持 `Copy`（供 MCTS 以值语义保存快照）的关键约束。
- 变体（4x4 / 4x2）仅复用 `DarkChessEnv` 内核 + 不同 `GameConfig`：包装类型只定义 `struct X { inner: DarkChessEnv }`，其模块级常量、固有委托方法、`Default` 与 `GameEnv` 委托实现全部由 `variants::impl_darkchess_variant!` 宏生成；常量单一真源为变体的 const fn 配置函数（如 `game_4x4_config()`）。

### 4.2 变体与动作空间

| 变体 | 构造 | 棋盘 | 激活棋子 | 动作空间 | 血量 | 判和 / 步数 |
|------|------|------|----------|----------|------|-------------|
| 4x8 暗棋 | `DarkChessEnv::new()` / `darkchess_config()` | 4×8=32 | 7 类（5/2/2/2/2/2/1） | 32+104+216 = **352** | 60 | 24 / 100 |
| 4x4 暗棋 | `DarkChessEnv::new_4x4()` / `game_4x4_config()` | 4×4=16 | 7 类（2/1/1/1/1/1/1） | 16+48+48 = **112** | 60 | 16 / 48 |
| 4x2 迷你 | `DarkChessEnv::new_mini()` / `mini_config()` | 4×2=8 | 兵/炮/士/将（1/1/0/0/0/1/1） | 8+20+12 = **40** | 47 | 8 / 30 |
| 井字棋 | `TicTacToeEnv::new()` | 3×3 | — | **9** | 无 | — |

动作空间组成（按此顺序拼接）：`翻棋`（每格一个）→ `常规移动`（四方向有效出边）→ `炮击`（同行/列距离 > 1 的有序对，去重）。逻辑由 `compute_action_counts` 与 `actions.rs::build_action_lookup_tables` **双重保证一致**。

### 4.3 `GameEnv` trait（MCTS 的接口契约）

`env::traits::GameEnv` 是 MCTS 泛型化的唯一约束，要求 `Copy + Clone + Send + Sync + 'static`：

- 无关联形状常量：棋盘行列 / 通道数 / 标量数 / 步数上限均由运行时 `config` 决定（不设编译期变体常量）；
- 核心方法：`action_space_size(&self)`、`get_current_player()`、`action_masks_into()`、`step() -> Result<..., EnvError>`、`check_game_over_conditions()`、`max_steps(&self)`、`encode_resnet_features_flat_into()`；`get_resnet_state()` 为必需方法（`DarkChessEnv` 由 `config` 推导形状，变体转发 inner，`TicTacToeEnv` 用自身常量）；
- `step` 返回结构化 `EnvError`：非法/越界动作为 `IllegalAction`，内部不变量破坏（源格非明子、阵亡池溢出、翻子后仍为暗子、隐藏池与指定棋子不匹配）为 `BrokenInvariant`；环境层热路径不再以 `panic!`/`expect` 中断搜索；
- **机会节点扩展点**（暗棋特有，默认关闭）：`is_chance_action()` / `chance_outcomes()` / `step_outcome_id()`。`DarkChessEnv`、`Game4x4Env`、`MiniDarkChessEnv` 覆盖实现；`TicTacToeEnv` 保持默认（无机会节点）；
- 终局血量辅助：`terminal_health_diff_red()` / `terminal_health_diff_red_int()` / `health_diff_scale()`，供 MCTS 血量复合效用使用。

### 4.4 `DarkChessEnv` 状态模型

`board/struct_def.rs` 定义，核心字段：

- 棋盘：`board: [Slot; 32]`（权威展示态）+ 位棋盘冗余索引；
- 位棋盘：`piece_bitboards[2][7]`、`revealed_bitboards[2]`、`hidden_bitboard`、`empty_bitboard`；
- 计数 / 统计：`move_counter`、`total_step_counter`、`scores[2]`（血量，吃子扣血）、`dead_pieces_pool/count`、`dead_piece_counts_by_type[2][7]`；
- 袋模型：`hidden_pieces_pool: [Piece; 32]` + `hidden_pieces_count` + `reveal_probabilities[14]`；
- 确定性：`seed`（`StdRng`）与 `true_board`（固定真实布局）。

### 4.5 `darkchess` 单步数据流

1. `step(action, reveal_piece)`：
   - 用 `action_masks_into` 校验合法性（非法返回 `Err`）；
   - `action < reveal_actions_count` → 走 `reveal_piece_at`（翻棋，重置 `move_counter`）；
   - 否则按 `action_to_coords` 取 `(from, to)` → `apply_move_action`（先处理目标格暗子翻出，再清除源/目标归属、写入攻击方；吃明子则记录阵亡 + 扣血）；
   - 切换 `current_player`，调用 `check_game_over_conditions()` 返回 `(terminated, truncated, winner)`（`reward` 恒为 `0.0`，胜负由 `winner` 表达）。
2. 终局判定（`rules.rs::check_game_over_with_moves`）优先级：血量归零 → 一方全灭 → 无合法走法 → 连续无吃子判和 → 步数上限截断。
3. 机会节点语义：目标格为 `Hidden` 时 `is_chance_action` 为真；`chance_outcomes` 枚举袋中所有可能棋子，按 `outcome_id`（紧凑类型索引 + 黑方偏移）分组给出 `(id, prob, next_env)`；`step_outcome_id` 由 `last_revealed_piece` 反推实际翻出结果。

### 4.6 特征编码

- **统一投影源** `StateView`（`features.rs`）：单次遍历内部状态，产出红黑绝对视角的只读快照；ResNet 与 NNUE 全部特征从这一份快照派生（将来状态重构只需改 `state_view`）。
- **ResNet 稠密特征** `get_resnet_state` / `resnet_features_flat_into`：
  - board 张量通道 = `2*num_active + 2`（己方各型明子 → 敌方各型明子 → 暗子 → 空位）；按**当前行棋方视角**排列；
  - scalars = `3 + 4*total_pieces`（`move_counter` 归一化 + 双方 HP 归一化 + 存活向量 ×2 + 暗子向量 ×2）。
- **NNUE 稀疏特征** `nnue_active_features_into`：布局由 `GameConfig` 推导（`nnue_feature_dim()`，4x8 = 555）：
  - 格位段 `total_positions * states_per_square`：状态按**相对行棋方视角**编码（0=空 / 1=暗 / 己方明子 / 对方明子，无红黑身份）；
  - 暗子包段 `num_active * bag_stride`：每型计数桶（红黑合并，归属不可见）；
  - 标量段 1 个：无吃子标记。

### 4.7 预计算表与缓存

- **统一缓存设施** `env/cache.rs`：`global_cache!(CACHE, getter, ValType)` 声明 `OnceLock<Mutex<HashMap<u64, Arc<T>>>>` 全局缓存，`cached(cache, key, build)` 完成查/建/插（构建在锁外进行）。
- `actions.rs`：`ACTION_TABLE_CACHE`，键为 `(rows, cols)`；`action_to_coords` / `coords_to_action`（`pack_coords` 编解码）；构建末尾 `debug_assert` 校验动作表长度与 `config.action_space_size` 一致。
- `bitboard.rs`：`RAY_CACHE`，键为 `(rows, cols)`；`ray_attacks` 提供四方向射线，用于炮击。
- `symmetry.rs`：`PERM_CACHE` / `SQMAP_CACHE`，`sq_map` / `action_permutation` / `transform_board_flat` / `transform_action`，语义与 Python `data_augmentation.py` 对齐。
- 所有缓存均以 `(rows, cols)` 分键，支持多变体在同进程共存。

---

## 5. MCTS 层（`core::mcts`）详解

### 5.1 定位

Gumbel AlphaZero 风格的 MCTS，泛型化于 `G: GameEnv`，通过 `Evaluator<G>` 对接上层神经网络。核心特性：Gumbel-Top-K 采样候选 + Sequential Halving 预算淘汰 + 同步批量评估（无异步/递归）。

### 5.2 一次 `run()` 的数据流

1. `expand_root()`：评估根环境，`compute_probs_from_logits` 归一化先验，`build_children_from_eval` 建子节点（继承父评估值作为 `initial_value`/`initial_health` 先验）。
2. 读根动作掩码；全 0 则返回 `None`。
3. `sample_gumbel_top_k(logits, mask, max_considered_actions)` 选出候选动作（候选数 ≤ 1 时直接构造结果返回）。
4. 构造 `SequentialHalvingBudget`，逐阶段循环：
   - 对本阶段每个候选调用 `select_path_collect(action, &mut batch)`：
     - 机会节点未展开 → `expand_chance_node` 全量展开并批量产出所有 outcome 的 `PendingEval`（带 `ChanceSeed`）；已展开 → 按概率采样一个 outcome 继续下探；
     - 终局节点 → 直接回传（`TerminalBackprop`）；
     - 未展开普通叶 → 收集进 `batch`；
     - 已展开普通节点 → 用 PUCT（`Q + c_scale·P·√N/(1+N)`）选子节点下探。
   - 合并 `batch` 送入 `evaluator.evaluate`，回填 logits/value/health → `build_children_from_eval` → `backprop_evals`。
   - 按 `completed_utility`（血量为开时 = Q_win + λ(|Q_win|)·Q_hp，否则 = Q_win）排序淘汰，`budget.keep_count_after_phase()` 决定保留数。
5. 剩余候选的 `[0]` 即动作；返回 `MctsSearchResult`（含 `get_improved_policy()` 训练目标 = `softmax(logit + σ·Q)`，σ = `c_scale·ln(1+N_root)`）。

### 5.3 模块职责分层

| 文件 | 职责 |
|------|------|
| `search.rs` | 搜索器结构体定义 + 构造 + 树推进（`step_next` / `root_env`） |
| `root.rs` | 根侧逻辑（run 与 batched 共用）：`expand_root` / `refresh_root_action_mask` / `root_logits` / `sample_root_candidates` / `apply_root_eval` / `apply_leaf_evals` / `completed_q` / `completed_utility` / `build_result` |
| `path_select.rs` | 路径选择 `select_path_collect`（机会节点处理、终局回传、步数上限兜底、PUCT 子节点选择） |
| `run.rs` | 主循环 `run`（Sequential Halving 编排：阶段执行 / 批量评估回填 / 候选淘汰 / 空转告警 / 终局动作） |
| `tree.rs` | 树构建与回溯（`build_children_from_eval` / `expand_chance_node` / `backprop_from_path` / `backprop_evals` / `node_q_value` / `node_utility_value`） |
| `policy.rs` | 只读策略计算（`compute_probs_from_logits` / `get_root_probabilities` / `get_improved_policy`） |
| `sampling.rs` | Gumbel Top-K / 机会结果采样 |
| `node.rs` | `MctsArena`（Slab）+ `MctsNode` + `value_from_perspective` |
| `path.rs` | 路径 / 待评估类型定义（无逻辑） |
| `budget.rs` | Sequential Halving 预算排程 |
| `batched.rs` | `BatchedTree`，多树 lockstep 合并大 batch，供自对弈吞吐优化 |

### 5.4 结构与语义约束（务必保留，见 `mcts/mod.rs` 顶部注释）

- **不要在机会节点改为非全量展开**；
- **不要移除"显式判断父子节点玩家是否一致以决定价值取反"**（`value_from_perspective` 的调用点）；
- 根节点 Dirichlet 噪声**已移除**，请勿加回（Gumbel 探索已足够，prior 不参与根决策）；
- 根节点温度采样**已移除**，请勿加回（落子直接用 `search_result.action`，避免与 Gumbel 噪声重复随机源、污染自对弈数据）。

### 5.5 批量自对弈（`batched.rs`）

`BatchedTree` 用 `Root / Searching / Ready / Idle` 状态机驱动单棵树：外部协调器对所有树 `collect` 叶子 → 合并成一个 batch → 单次 `evaluator.evaluate` → `apply` 回填；每步 `finalize_step` 产出动作并推进，`start_next_step` 开启下一步。任一树在叶子被评估/回填前不前进（严格 lockstep），保证路径不漂移。

---

## 6. Expectimax 层（`core::expectimax`）详解

### 6.1 定位与值约定

Star1 概率节点剪枝的 Expecti-Alpha-Beta 强搜索，绑定 `DarkChessEnv`（非泛型）。启用的技术：置换表、走子排序、静态搜索（quiescence）、LMR、迭代加深、Lazy SMP、血量复合效用。

- **值约定**：所有搜索值均为"当前节点走子方视角"，范围约 `[-1, 1]`（`VMIN=-1` / `VMAX=1`）。
- **叶评估唯一来源**：NNUE。`SearchConfig.nnue_evaluator = None` 时 `search` / `search_par` **直接拒绝执行**并打印错误（不提供规则评估兜底），并在入口用 `validate_feature_dim(env.config.nnue_feature_dim())` 做强校验。

### 6.2 入口与配置

- `ExpectimaxEngine::with_nnue(Arc<dyn NnueEvaluate>)` 创建引擎；`search` / `search_par` / `best_action`；
- `SearchConfig` 关键字段：`node_budget` / `time_limit_ms` / `max_depth` / `contempt` / `quiesce(+_max)` / `features`（`FEAT_*`）/ `tt_bits` / `threads` / `chance_reduction` / `tt_sym_probe` / `nnue_evaluator`；
- `search_par`：`threads <= 1` 时等价 `search`；`>1` 时主线程迭代加深，助线程独立迭代加深仅通过共享 `SharedTT` 贡献信息，**主线程结果为准**。

### 6.3 主搜索数据流（`search.rs`）

- `negamax(env, acc, depth, α, β)`：
  1. `generate_moves` → `terminal_info` 终局短路（`terminal_value` 含 contempt 偏差）；
  2. 深度耗尽 → `quiesce`（若开启）否则 `eval_acc`；
  3. 计算 `zkey`，重复检测（`FEAT_REP`，命中判和）；
  4. TT 探测（`FEAT_TT`；`tt_sym_probe` 时 miss 后用对称键二次探测，仅统计）；
  5. 走子排序 + TT move 提前；
  6. 遍历走子：`move_value`（机会动作走 `flip_value`，否则 step 后 `-negamax`）；满足条件时用 LMR 先试探再全搜；
  7. 静走截断 → `record_cutoff`（杀手 + 历史）；
  8. TT 条件存储（exact / 下界 / 上界）。
- `flip_value`（Star1 机会节点）：枚举 `chance_outcomes`，按概率加权期望，用区间边界 `u/l` 做剪枝；`chance_reduction` 对子分支额外减深。
- `best_at_depth` + `search_with_tt`：根层单层搜索 + 迭代加深（1..=max_depth），维护 `hint` 用于下一层排序。

### 6.4 辅助模块

- `nnue.rs`：`NnueEvaluate`（`evaluate` / `validate_feature_dim` / `init_accumulator`）与 `NnueAccumulator`（`clone_box` / `apply_step` / `evaluate(player)`）契约；具体量化网络实现位于上层 crate；测试用 `DummyNnue`。
- `ordering.rs`：`order_key` 优先级 = 吃子 MVV-LVA > 杀手 > 历史静走 > 炮吃暗子（机会）> 翻棋垫底；MVV-LVA 的棋子分值统一取 `GameConfig.piece_values`（与运行时扣血同源）；`terminal_value` / `terminal_info` / `victim_value`。
- `zobrist.rs`：`zkey` = 棋盘槽位 + 暗子袋（按颜色/类型计数）+ 走子方；**和棋时钟不参与哈希**（换取更多 TT 命中）；`sym_zkey` 为对称视角键；随机数用 SplitMix64，跨运行确定。
- `smp.rs`：`SharedTT` 将表项打包为单个 `AtomicU64`（value | depth | flag | key_check），best 提示独立 `AtomicU32`；写入 last-write-wins。

实现分文件（`search.rs` 为入口）：`negamax.rs`（`negamax` / `quiesce` / `flip_value` / `move_value` / TT 探测与存储 / `search_ordered_moves`）、`iterative.rs`（`best_at_depth` / `iterative_deepen` / `move_to_front`）、`eval.rs`（`eval_state` / `eval_acc` / `step_with_acc` / `outcome_acc`）、`config.rs`（`SearchConfig` / `SearchResult`，经 `search` 再导出）。

---

## 7. 模块依赖与对接关系

```
上层 crate（banqi_4x8 / banqi-engine / banqi-tauri / Python 绑定）
        │  实现 trait
        ▼
  core::mcts::Evaluator<G> ─────► core::mcts（GumbelMCTS / BatchedTree）
  core::expectimax::nnue::NnueEvaluate ─► core::expectimax（ExpectimaxEngine）
        │
        ▼
  core::env::GameEnv ◄── DarkChessEnv / Game4x4Env / MiniDarkChessEnv / TicTacToeEnv
        │
        ├── config（GameConfig / 变体配置）
        ├── rules（终局 / 动作掩码 / generate_moves）
        ├── features（StateView → ResNet / NNUE）
        ├── actions / bitboard / symmetry（config 驱动预计算 + 缓存）
        └── board（struct_def / reset / step / accessors）

core::expectimax ──依赖──► core::env（DarkChessEnv、symmetry）
core::mcts ────────依赖──► core::env（GameEnv、ResNetObservation、Player）
core::mcts 与 core::expectimax 之间无相互依赖（各自独立搜索栈，共享 env 与 zobrist 语义）
```

---

## 8. 快速定位索引

| 我想…… | 去看 |
|--------|------|
| 改游戏规则 / 终局判定 / 动作合法性 | `env/rules.rs` |
| 改单步行为 / 吃子扣血 / 翻棋 | `env/board/step.rs` |
| 加新变体 / 调子力 / 调血量 / 调动作空间 | `env/config.rs` + `env/variants/*` |
| 改网络输入特征 | `env/features.rs`（同时看 `GameConfig` 布局推导） |
| 改 MCTS 搜索 / 候选采样 / 淘汰逻辑 | `mcts/search.rs`、`mcts/sampling.rs`、`mcts/budget.rs` |
| 改 MCTS 树构建 / 回溯 / 机会节点处理 | `mcts/tree.rs`、`mcts/node.rs`、`mcts/path.rs` |
| 改 MCTS 训练目标策略 | `mcts/policy.rs` |
| 改 Expectimax 搜索 / 剪枝 / TT | `expectimax/search.rs`、`expectimax/smp.rs` |
| 改走子排序 | `expectimax/ordering.rs` |
| 改局面哈希 / 对称 | `expectimax/zobrist.rs`、`env/symmetry.rs` |
| 对接神经网络 | `mcts/evaluator.rs`、`expectimax/nnue.rs` |
| 数据增强 / 动作置换 | `env/symmetry.rs` |
