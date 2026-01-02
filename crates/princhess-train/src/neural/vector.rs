use crate::neural::activation::Activation;
use bytemuck::Zeroable;
use std::ops::{Add, AddAssign, Deref, Div, DivAssign, Index, IndexMut, Mul, MulAssign, SubAssign};

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseVector {
    inner: Vec<usize>,
}

impl Add<SparseVector> for SparseVector {
    type Output = SparseVector;
    fn add(mut self, mut rhs: SparseVector) -> Self::Output {
        self.inner.append(&mut rhs.inner);
        self
    }
}

impl Deref for SparseVector {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl SparseVector {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            inner: Vec::with_capacity(cap),
        }
    }

    pub fn push(&mut self, val: usize) {
        self.inner.push(val);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Zeroable)]
pub struct Vector<const N: usize> {
    inner: [f32; N],
}

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

impl<const N: usize> Add<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn add(mut self, rhs: Vector<N>) -> Self::Output {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i += *j;
        }

        self
    }
}

impl<const N: usize> Add<f32> for Vector<N> {
    type Output = Vector<N>;
    fn add(mut self, rhs: f32) -> Self::Output {
        for i in self.inner.iter_mut() {
            *i += rhs;
        }

        self
    }
}

impl<const N: usize> AddAssign<Vector<N>> for Vector<N> {
    fn add_assign(&mut self, rhs: Vector<N>) {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i += *j;
        }
    }
}

impl<const N: usize> DivAssign<f32> for Vector<N> {
    fn div_assign(&mut self, rhs: f32) {
        for x in self.inner.iter_mut() {
            *x /= rhs;
        }
    }
}

impl<const N: usize> MulAssign<f32> for Vector<N> {
    fn mul_assign(&mut self, rhs: f32) {
        for x in self.inner.iter_mut() {
            *x *= rhs;
        }
    }
}

impl<const N: usize> Div<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn div(mut self, rhs: Vector<N>) -> Self::Output {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i /= *j;
        }

        self
    }
}

impl<const N: usize> Mul<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn mul(mut self, rhs: Vector<N>) -> Self::Output {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i *= *j;
        }

        self
    }
}

impl<const N: usize> Mul<Vector<N>> for f32 {
    type Output = Vector<N>;
    fn mul(self, mut rhs: Vector<N>) -> Self::Output {
        for i in rhs.inner.iter_mut() {
            *i *= self;
        }

        rhs
    }
}

impl<const N: usize> SubAssign<Vector<N>> for Vector<N> {
    fn sub_assign(&mut self, rhs: Vector<N>) {
        for (i, j) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *i -= *j;
        }
    }
}

impl<const N: usize> Vector<N> {
    pub fn from_fn<F: FnMut(usize) -> f32>(mut f: F) -> Self {
        let mut res = Self::zeroed();

        for i in 0..N {
            res.inner[i] = f(i);
        }

        res
    }

    pub fn dot(&self, other: &Vector<N>) -> f32 {
        let mut score = 0.0;
        for (&i, &j) in self.inner.iter().zip(other.inner.iter()) {
            score += i * j;
        }

        score
    }

    pub fn out<T: Activation>(&self, other: &Vector<N>) -> f32 {
        let mut score = 0.0;
        for (i, j) in self.inner.iter().zip(other.inner.iter()) {
            score += T::activate(*i) * T::activate(*j);
        }

        score
    }

    pub fn sqrt(mut self) -> Self {
        for i in self.inner.iter_mut() {
            *i = i.sqrt();
        }

        self
    }

    pub const fn from_raw(inner: [f32; N]) -> Self {
        Self { inner }
    }

    pub const fn zeroed() -> Self {
        Self::from_raw([0.0; N])
    }

    pub fn activate<T: Activation>(mut self) -> Self {
        for i in self.inner.iter_mut() {
            *i = T::activate(*i);
        }

        self
    }

    pub fn derivative<T: Activation>(mut self) -> Self {
        for i in self.inner.iter_mut() {
            *i = T::derivative(*i);
        }

        self
    }

    pub fn madd(&mut self, other: &Self, mul: f32) {
        for (i, j) in self.inner.iter_mut().zip(other.inner.iter()) {
            *i += mul * *j;
        }
    }
}

impl<const N: usize> Div<f32> for Vector<N> {
    type Output = Self;

    fn div(mut self, rhs: f32) -> Self::Output {
        for i in self.inner.iter_mut() {
            *i /= rhs;
        }
        self
    }
}
