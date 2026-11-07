use std::mem;
use std::ops::{Deref, DerefMut};

use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Zeroable)]
#[repr(C, align(64))]
pub struct Align64<T>(pub T);

unsafe impl<T: Pod> Pod for Align64<T> {}

impl<T> Deref for Align64<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align64<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub const HUGE_PAGE_SIZE: usize = 2 << 20;

#[repr(C, align(2097152))]
pub struct AlignHugePage<T>(pub T);

const _: () = assert!(mem::align_of::<AlignHugePage<u8>>() == HUGE_PAGE_SIZE);

impl<T> Deref for AlignHugePage<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for AlignHugePage<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(target_os = "linux")]
pub fn advise_huge_pages(ptr: *mut u8, len: usize) {
    // SAFETY: madvise only changes paging hints. It's best effort, so a failure is ignored.
    unsafe {
        libc::madvise(ptr.cast(), len, libc::MADV_HUGEPAGE);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn advise_huge_pages(_ptr: *mut u8, _len: usize) {}

#[derive(Debug, Clone, Copy, Zeroable)]
#[repr(C, align(16))]
pub struct Align16<T>(pub T);

unsafe impl<T: Pod> Pod for Align16<T> {}

impl<T> Deref for Align16<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align16<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
