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
