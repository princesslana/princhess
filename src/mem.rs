use std::alloc::{self, Layout};
use std::boxed::Box;
use std::ops::{Deref, DerefMut};

pub fn boxed_and_zeroed<T>() -> Box<T> {
    unsafe {
        let layout = Layout::new::<T>();
        let ptr = alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Box::from_raw(ptr.cast())
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct Align64<T>(pub T);

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

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct Align16<T>(pub T);

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
