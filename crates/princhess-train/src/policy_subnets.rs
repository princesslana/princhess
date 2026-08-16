use std::ops::{AddAssign, DivAssign};

use bytemuck::Zeroable;
use princhess::chess::Square;
use princhess::state::POLICY_NUMBER_FEATURES;

use crate::neural::{AdamWOptimizer, LRScheduler, LinearNetwork};

pub type PolicyLinearNetwork<const A: usize> = LinearNetwork<POLICY_NUMBER_FEATURES, A>;

#[derive(Zeroable)]
pub struct SquareSubnets<const A: usize>(pub [PolicyLinearNetwork<A>; Square::COUNT]);

impl<const A: usize> std::ops::Deref for SquareSubnets<A> {
    type Target = [PolicyLinearNetwork<A>; Square::COUNT];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const A: usize> std::ops::DerefMut for SquareSubnets<A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const A: usize> AddAssign<&Self> for SquareSubnets<A> {
    fn add_assign(&mut self, rhs: &Self) {
        for (l, r) in self.0.iter_mut().zip(&rhs.0) {
            *l += r;
        }
    }
}

impl<const A: usize> DivAssign<f32> for SquareSubnets<A> {
    fn div_assign(&mut self, rhs: f32) {
        for s in &mut self.0 {
            *s /= rhs;
        }
    }
}

impl<const A: usize> SquareSubnets<A> {
    pub fn l1_norm(&self) -> f32 {
        self.0.iter().map(PolicyLinearNetwork::<A>::l1_norm).sum()
    }

    pub fn adamw<S: LRScheduler>(
        &mut self,
        g: &Self,
        m: &mut Self,
        v: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    ) {
        for i in 0..Square::COUNT {
            self.0[i].adamw(&g.0[i], &mut m.0[i], &mut v.0[i], optimizer);
        }
    }

    pub fn randomize(&mut self) {
        for s in &mut self.0 {
            s.randomize();
        }
    }
}

#[derive(Zeroable)]
pub struct SeeSplitSubnets<const A: usize> {
    pub base: SquareSubnets<A>,
    pub good_see: SquareSubnets<A>,
}

impl<const A: usize> AddAssign<&Self> for SeeSplitSubnets<A> {
    fn add_assign(&mut self, rhs: &Self) {
        self.base += &rhs.base;
        self.good_see += &rhs.good_see;
    }
}

impl<const A: usize> DivAssign<f32> for SeeSplitSubnets<A> {
    fn div_assign(&mut self, rhs: f32) {
        self.base /= rhs;
        self.good_see /= rhs;
    }
}

impl<const A: usize> SeeSplitSubnets<A> {
    pub fn l1_norm(&self) -> f32 {
        self.base.l1_norm() + self.good_see.l1_norm()
    }

    pub fn adamw<S: LRScheduler>(
        &mut self,
        g: &Self,
        m: &mut Self,
        v: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    ) {
        self.base
            .adamw(&g.base, &mut m.base, &mut v.base, optimizer);
        self.good_see
            .adamw(&g.good_see, &mut m.good_see, &mut v.good_see, optimizer);
    }

    pub fn randomize(&mut self) {
        self.base.randomize();
        self.good_see.randomize();
    }
}
