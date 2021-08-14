use std;

#[cfg(not(any(target_pointer_width = "64", feature = "nightly")))]
compile_error!("If you aren't compiling for 64-bit, you must use the nightly compiler.");

#[cfg(target_pointer_width = "64")]
pub type AtomicI64 = std::sync::atomic::AtomicIsize;
#[cfg(not(target_pointer_width = "64"))]
pub type AtomicI64 = std::sync::atomic::AtomicI64;

#[cfg(target_pointer_width = "64")]
pub type AtomicU64 = std::sync::atomic::AtomicUsize;
#[cfg(not(target_pointer_width = "64"))]
pub type AtomicU64 = std::sync::atomic::AtomicU64;

#[cfg(target_pointer_width = "64")]
pub type FakeU64 = usize;
#[cfg(not(target_pointer_width = "64"))]
pub type FakeU64 = u64;

#[cfg(target_pointer_width = "64")]
pub type FakeI64 = isize;
#[cfg(not(target_pointer_width = "64"))]
pub type FakeI64 = i64;

// #[cfg(feature = "nightly")]
// pub type FakeU32 = std::sync::atomic::AtomicU32;
// #[cfg(not(feature = "nightly"))]
// pub type FakeU32 = std::sync::atomic::AtomicUsize;

pub type FakeU32 = std::sync::atomic::AtomicU32;

pub type AtomicPtr<T> = std::sync::atomic::AtomicPtr<T>;
pub type AtomicBool = std::sync::atomic::AtomicBool;
pub type AtomicIsize = std::sync::atomic::AtomicIsize;
pub type AtomicUsize = std::sync::atomic::AtomicUsize;
pub use std::sync::atomic::Ordering;
