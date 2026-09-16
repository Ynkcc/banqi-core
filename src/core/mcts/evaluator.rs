// src/mcts/evaluator.rs
// 神经网络评估接口定义（泛型化：G = 游戏环境）

use std::fmt;

use crate::core::env::GameEnv;

/// 批量评估失败：推理后端报错，或模型输出不符合前向契约（形状/输出结构）。
///
/// `evaluate` 返回 `Result` 而非直接 panic / 退化为均匀策略：
/// 推理失败必须显式向上传播，由调用方决定重试或终止本批次；
/// 静默退化会让自对弈数据质量无声下降。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatorError(String);

impl EvaluatorError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EvaluatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "评估失败: {}", self.0)
    }
}

impl std::error::Error for EvaluatorError {}

impl From<String> for EvaluatorError {
    fn from(message: String) -> Self {
        Self(message)
    }
}

impl From<&str> for EvaluatorError {
    fn from(message: &str) -> Self {
        Self(message.to_string())
    }
}

/// 评估器输出：统一承载策略 logits、胜率与可选的「血量差异分桶 logits」。
///
/// `health` 仅在模型带血量差异头时返回 `Some([B, K])`，K = 2*INITIAL_HEALTH+1；
/// 否则为 `None`（旧模型 / 启发式评估器等）。MCTS 是否使用血量由
/// `GumbelConfig.health_enabled` 决定，二者独立解耦。
pub struct EvaluatorOutput {
    /// 每个环境的动作原始 Logits（未 mask/softmax）
    pub logits: Vec<Vec<f32>>,
    /// 每个环境的状态价值（胜率，[-1, 1]）
    pub values: Vec<f32>,
    /// 每个环境的血量差异分桶 Logits（[B, K]）；无血量头时为 None
    pub health: Option<Vec<Vec<f32>>>,
}

impl EvaluatorOutput {
    /// 从第 `idx` 个环境的血量分桶 logits 计算期望血量差 μ ∈ [-1, 1]。
    ///
    /// 分桶定义：桶 i 的整数中心 c_i = i - D，D = (K-1)/2（D = INITIAL_HEALTH），
    /// 归一化中心 = c_i / D。μ = Σ softmax(logits)_i · (i - D)/D。
    /// health 缺失或 K<3 时返回 None。
    pub fn health_expectation(&self, idx: usize) -> Option<f32> {
        health_logits_expectation(self.health.as_deref(), idx)
    }

    /// 取血量期望，并校验「血量开关与评估器能力一致」（Part C #5）。
    ///
    /// `required=true`（= `GumbelConfig::health_active()`）而评估器没给出 health 输出
    /// （模型只有 policy/value 两头，或后端不支持）时**返回 Err 而非 0**：静默填 0 会让
    /// 血量项恒为 0 且无任何提示，H2 类实验会在「看起来正常」的表象下得出错误结论。
    pub fn health_expectation_required(
        &self,
        idx: usize,
        required: bool,
    ) -> Result<f32, EvaluatorError> {
        health_expectation_required(self.health.as_deref(), idx, required)
    }
}

/// 取血量期望，并按 `required` 校验评估器能力（见
/// [`EvaluatorOutput::health_expectation_required`]）。供批量自对弈路径复用——
/// 该路径不经过 `EvaluatorOutput` 载体，只拿得到 `Option<&[Vec<f32>]>`。
pub fn health_expectation_required(
    health: Option<&[Vec<f32>]>,
    idx: usize,
    required: bool,
) -> Result<f32, EvaluatorError> {
    let mu = health_logits_expectation(health, idx);
    if required && mu.is_none() {
        return Err(EvaluatorError::new(
            "GumbelConfig.health_active()=true，但评估器未提供血量分桶输出（模型只有 \
             policy/value 两头）——血量项会静默恒 0 并污染实验结论。请改用带血量头的 \
             模型，或关闭 health_enabled/health_weight",
        ));
    }
    Ok(mu.unwrap_or(0.0))
}

/// 由血量分桶 logits 计算期望血量差 μ ∈ [-1, 1]。
///
/// 独立的自由函数，供批处理自对弈（非 `EvaluatorOutput` 载体）复用。
pub fn health_logits_expectation(health: Option<&[Vec<f32>]>, idx: usize) -> Option<f32> {
    let row = health?.get(idx)?;
    let k = row.len();
    if k < 3 {
        return None;
    }
    let d = ((k - 1) / 2) as f32; // D = (K-1)/2
    if d <= 0.0 {
        return None;
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let mut sum = 0.0f32;
    let mut num = 0.0f32;
    for (i, &x) in row.iter().enumerate() {
        let w = (x - max).exp();
        sum += w;
        num += w * ((i as f32) - d);
    }
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    Some((num / sum / d).clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot_logits(k: usize, idx: usize) -> Vec<Vec<f32>> {
        let mut row = vec![f32::NEG_INFINITY; k];
        row[idx] = 0.0; // softmax 下仅该桶概率为 1
        vec![row]
    }

    #[test]
    fn health_expectation_onehot_bin_centers() {
        // K = 2D+1 => D = (K-1)/2；one-hot 于桶 i => μ = (i - D)/D
        for k in [121usize, 95usize] {
            let d = ((k - 1) / 2) as f32;
            for i in [0usize, (k - 1) / 2, k - 1] {
                let mu = health_logits_expectation(Some(&one_hot_logits(k, i)), 0).unwrap();
                let expect = (i as f32 - d) / d;
                assert!((mu - expect).abs() < 1e-4, "k={k} i={i} mu={mu} expect={expect}");
            }
        }
    }

    #[test]
    fn health_expectation_softmax_mixing() {
        // 两桶各 0.5 概率，期望应为两中心均值
        let k = 5; // D=2，中心 -2..2
        let mut row = vec![f32::NEG_INFINITY; k];
        row[0] = 0.0; // 中心 -2
        row[4] = 0.0; // 中心 +2
        let mu = health_logits_expectation(Some(&[row]), 0).unwrap();
        assert!((mu - 0.0).abs() < 1e-4, "mu={mu}");
    }

    #[test]
    fn health_expectation_missing_returns_none() {
        assert_eq!(health_logits_expectation(None, 0), None);
        assert_eq!(health_logits_expectation(Some(&vec![vec![0.0f32]]), 0), None);
        // 非有限 logits（全 -inf）也返回 None
        let bad = vec![vec![f32::NEG_INFINITY; 121]];
        assert_eq!(health_logits_expectation(Some(&bad), 0), None);
    }
}

/// 评估器特征 (Trait)
///
/// 定义了评估游戏状态的接口。
/// 实现该特征的结构体 (如神经网络模型) 需要提供状态评估功能。
pub trait Evaluator<G: GameEnv> {
    /// 评估给定的游戏环境批次
    ///
    /// # 参数
    ///
    /// * `envs` - 需要评估的 `G` 列表
    ///
    /// # 返回
    ///
    /// 返回 `EvaluatorOutput`：
    /// * `logits`: 每个环境的动作原始 Logits（未 mask/softmax）
    /// * `values`: 每个环境的状态价值
    /// * `health`: 每个环境的血量差异分桶 Logits（可选）
    ///
    /// 推理失败或输出不符合契约时返回 `Err(EvaluatorError)`，不得 panic，
    /// 也不得静默返回退化结果。
    fn evaluate(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError>;

    /// 评估并返回 Logits 和 Value
    ///
    /// 默认实现直接返回 `evaluate` 的结果。
    /// Logits 用于 Gumbel 分布的采样。
    fn evaluate_logits(&self, envs: &[G]) -> Result<EvaluatorOutput, EvaluatorError> {
        self.evaluate(envs)
    }
}
