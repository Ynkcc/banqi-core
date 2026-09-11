# ARCHITECTURE — banqi-core 领域核心

## 定位

Banqi（暗棋）领域核心 crate：游戏环境、Gumbel MCTS、Expectimax 强引擎、NNUE 量化推理。
零 torch/onnx/pyo3 依赖，可独立编译、独立发布。

> 迁移自 `4x8` 仓库 `crates/banqi-core/`（拆分规划见原仓库 `docs/ARCHITECTURE.md` §2.0）。
> `Evaluator` trait（`core/mcts/evaluator.rs`）定义于此；上层（TorchScript / ONNX / PyEvaluator）为实现方。

## 模块

- `core/env/`：暗棋环境——types/config/constants/actions/rules/bitboard/features/symmetry/seed（`AsDarkChessRef`/`SeedableEnv`）/board（`DarkChessEnv`）/variants（4x8、4x4、mini、井字棋）
- `core/mcts/`：Gumbel MCTS（tree/node/search/policy/sampling/batched/budget/evaluator/config）
- `core/expectimax/`：Star1 机会节点剪枝 + 共享 TT + LMR + 静态搜索 + 迭代加深 + Lazy SMP
- `core/zobrist.rs`：Zobrist 哈希
- `engine/movegen/`：走子生成
- `inference/nnue/`：增量累加器（`Accumulator`/`DualAccumulator`）+ 量化网络前向（`NnueEvaluator`）

## 构建

```bash
cargo build
cargo test
```

## 与主仓库的关系

`4x8` 主仓库当前以 path 依赖引用本 crate（`crates/banqi-core` 副本）；主仓库切换为 git 依赖后，`crates/banqi-core` 将删除。
