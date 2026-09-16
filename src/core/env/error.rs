use std::fmt;

/// 游戏环境执行动作时的错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    /// 动作不在当前合法动作集合内（含越界）。
    IllegalAction { action: usize },
    /// 内部不变量被破坏（走子源格非明子 / 阵亡池溢出 / 翻子后仍为暗子等）。
    /// 出现即为代码缺陷，携带出错位置以便定位。
    BrokenInvariant { context: &'static str },
    /// 局面快照载荷非法（跨进程传输的字节串损坏 / 版本不符 / 与变体不符）。
    /// 与 `BrokenInvariant` 区分：这是**输入数据**问题，不是代码缺陷。
    InvalidSnapshot { context: &'static str },
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnvError::IllegalAction { action } => write!(f, "无效动作: {action}"),
            EnvError::BrokenInvariant { context } => {
                write!(f, "环境内部状态异常: {context}")
            }
            EnvError::InvalidSnapshot { context } => {
                write!(f, "局面快照非法: {context}")
            }
        }
    }
}

impl std::error::Error for EnvError {}
