use crate::neural::Vector;
use bytemuck::Zeroable;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Zeroable)]
pub struct Matrix<const M: usize, const N: usize> {
    inner: [Vector<N>; M],
}

impl<const M: usize, const N: usize> std::ops::AddAssign<&Matrix<M, N>> for Matrix<M, N> {
    fn add_assign(&mut self, rhs: &Matrix<M, N>) {
        for (u, v) in self.inner.iter_mut().zip(rhs.inner.iter()) {
            *u += *v;
        }
    }
}

impl<const M: usize, const N: usize> std::ops::DivAssign<f32> for Matrix<M, N> {
    fn div_assign(&mut self, rhs: f32) {
        for row in self.inner.iter_mut() {
            *row /= rhs;
        }
    }
}

impl<const M: usize, const N: usize> std::ops::MulAssign<f32> for Matrix<M, N> {
    fn mul_assign(&mut self, rhs: f32) {
        for row in self.inner.iter_mut() {
            *row *= rhs;
        }
    }
}

impl<const M: usize, const N: usize> std::ops::Deref for Matrix<M, N> {
    type Target = [Vector<N>; M];
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<const M: usize, const N: usize> std::ops::DerefMut for Matrix<M, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<const M: usize, const N: usize> Matrix<M, N> {
    pub const fn zeroed() -> Self {
        Self::from_raw([Vector::zeroed(); M])
    }

    pub const fn from_raw(inner: [Vector<N>; M]) -> Self {
        Self { inner }
    }

    pub fn from_fn<F: FnMut(usize, usize) -> f32>(mut f: F) -> Self {
        let mut rows = [Vector::zeroed(); M];

        for (i, row) in rows.iter_mut().enumerate() {
            let inner_f = |j| f(i, j);
            *row = Vector::from_fn(inner_f);
        }

        Self::from_raw(rows)
    }

    pub fn mul(&self, inp: &Vector<M>) -> Vector<N> {
        let mut result = Vector::zeroed();

        for i in 0..M {
            result.madd(&self.inner[i], inp[i]);
        }

        result
    }

    pub fn transpose_mul(&self, out: &Vector<N>) -> Vector<M> {
        Vector::from_fn(|i| {
            let mut v = 0.0;
            for j in 0..N {
                v += self.inner[i][j] * out[j];
            }
            v
        })
    }
}
