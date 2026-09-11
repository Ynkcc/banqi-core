//! 全局 `OnceLock` 缓存：`key -> Arc<T>` 的查/建/插统一实现。
//!
//! 取代各模块手写的「加锁查缓存 → 构建 → insert → Arc::clone」样板。

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
