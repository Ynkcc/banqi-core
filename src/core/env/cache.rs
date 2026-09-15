//! 预计算表的进程内缓存，按使用场景分两种形态：
//!
//! - [`VariantCache`]：**热路径**专用（走法生成、特征编码每步都会调用）。按变体索引的
//!   `OnceLock` 数组，初始化后读路径只剩一次 acquire load（纯读，cache line 常驻
//!   Shared），返回 `&'static T` 也不引入引用计数。
//! - `global_cache!` + [`cached`]：**冷路径**（如数据增强的 D4 置换表，键是任意 u64）。
//!   每次查询要 `Mutex::lock`（RMW）并 `Arc::clone`（RMW），所有核心争用同一条
//!   cache line；放热路径上实测会把 12 线程的 CPU 吃掉约 1/3，故不再用于走法生成。

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::core::env::config::Variant;

pub(crate) type CacheMap<T> = Mutex<HashMap<u64, Arc<T>>>;

/// 声明一个以 u64 为键的全局 `OnceLock` 缓存及其 getter。
macro_rules! global_cache {
    ($cache:ident, $getter:ident, $val:ty) => {
        static $cache: std::sync::OnceLock<$crate::core::env::cache::CacheMap<$val>> =
            std::sync::OnceLock::new();
        pub(crate) fn $getter() -> &'static $crate::core::env::cache::CacheMap<$val> {
            $cache.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
        }
    };
}
pub(crate) use global_cache;

/// 查缓存，未命中则在锁外构建并插入（double-checked）。
/// 锁中毒说明曾有线程在持有该锁时 panic，状态不可信，应尽早暴露而非静默继续。
pub(crate) fn cached<T>(cache: &CacheMap<T>, key: u64, build: impl FnOnce() -> T) -> Arc<T> {
    {
        let guard = cache.lock().expect("cache lock poisoned: 持有锁的线程发生 panic");
        if let Some(t) = guard.get(&key) {
            return Arc::clone(t);
        }
    }
    let value = Arc::new(build());
    let mut guard = cache
        .lock()
        .expect("cache lock poisoned: 持有锁的线程发生 panic");
    guard.entry(key).or_insert(value).clone()
}

/// 按变体索引的只读表缓存（进程内每个变体一份，建好即只读）。
///
/// 变体集合由 `Variant` 枚举固定（3 个），表内容只由变体决定，故用定长 `OnceLock`
/// 数组代替 `Mutex<HashMap>`：读路径无锁、无哈希、无原子读改写。
pub(crate) struct VariantCache<T> {
    slots: [OnceLock<T>; Variant::ALL.len()],
}

impl<T> VariantCache<T> {
    pub(crate) const fn new() -> Self {
        Self {
            slots: [const { OnceLock::new() }; Variant::ALL.len()],
        }
    }

    /// 取该变体的表；首次调用时构建（并发首次访问由 `OnceLock` 保证只构建一次，
    /// 其余线程等待其结果，不会反复抢锁）。
    pub(crate) fn get(&self, variant: Variant, build: impl FnOnce() -> T) -> &T {
        debug_assert_eq!(
            Variant::ALL[variant.index()],
            variant,
            "Variant::index 与 Variant::ALL 的顺序不一致"
        );
        self.slots[variant.index()].get_or_init(build)
    }
}
