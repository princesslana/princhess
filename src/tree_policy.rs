extern crate rand;
use self::rand::rngs::SmallRng;
use self::rand::{Rng, SeedableRng};
use fastapprox::faster;

use mcts::MCTS;

use search_tree::*;
use state::State;
use std;

pub struct Fraction(pub f32, pub f32);

impl From<f32> for Fraction {
    fn from(x: f32) -> Fraction {
        Fraction(x, 1.)
    }
}

pub trait TreePolicy<Spec: MCTS<TreePolicy = Self>>: Sync + Sized {
    type ThreadLocalData: Default;

    fn choose_child<'a>(
        &self,
        state: &State,
        moves: Moves<'a>,
        handle: SearchHandle<Spec>,
    ) -> MoveInfoHandle<'a>;
}

#[derive(Clone, Debug)]
pub struct UctPolicy {
    cpuct: f32,
    cpuct_base: f32,
    cpuct_factor: f32,
}

impl UctPolicy {
    pub fn new(cpuct: f32, cpuct_base: f32, cpuct_factor: f32) -> Self {
        Self {
            cpuct,
            cpuct_base,
            cpuct_factor,
        }
    }
}

impl<Spec: MCTS<TreePolicy = Self>> TreePolicy<Spec> for UctPolicy {
    type ThreadLocalData = PolicyRng;

    fn choose_child<'a>(
        &self,
        _: &State,
        moves: Moves<'a>,
        mut handle: SearchHandle<Spec>,
    ) -> MoveInfoHandle<'a> {
        let total_visits = moves.map(|x| x.visits()).sum::<u64>();
        let adjusted_total = (total_visits + 1) as f32;
        let ln_adjusted_total = adjusted_total.ln();
        let exploration_constant = self.cpuct
            + self.cpuct_factor
                * faster::ln((total_visits as f32 + self.cpuct_base + 1.0) / self.cpuct_base);

        handle
            .thread_data()
            .policy_data
            .select_by_key(moves, |mov| {
                let sum_rewards = mov.sum_rewards();
                let child_visits = mov.visits();
                // http://mcts.ai/pubs/mcts-survey-master.pdf
                let explore_term = if child_visits == 0 {
                    std::f32::INFINITY
                } else {
                    2.0 * (ln_adjusted_total / child_visits as f32).sqrt()
                };
                let mean_action_value = sum_rewards as f32 / adjusted_total;
                (exploration_constant * explore_term + mean_action_value).into()
            })
            .unwrap()
    }
}

#[derive(Clone)]
pub struct PolicyRng {
    pub rng: SmallRng,
}

impl PolicyRng {
    pub fn new() -> Self {
        let rng = SeedableRng::seed_from_u64(42);
        Self { rng }
    }

    pub fn select_by_key<T, Iter, KeyFn>(&mut self, elts: Iter, mut key_fn: KeyFn) -> Option<T>
    where
        Iter: Iterator<Item = T>,
        KeyFn: FnMut(&T) -> Fraction,
    {
        let mut choice = None;
        let mut num_optimal: u32 = 0;
        let mut best_so_far: Fraction = std::f32::NEG_INFINITY.into();
        for elt in elts {
            let score = key_fn(&elt);
            let a = score.0 * best_so_far.1;
            let b = score.1 * best_so_far.0;
            if a > b {
                choice = Some(elt);
                num_optimal = 1;
                best_so_far = score;
            } else if (a - b).abs() < std::f32::EPSILON {
                num_optimal += 1;
                if self.rng.gen_bool(1.0 / num_optimal as f64) {
                    choice = Some(elt);
                }
            }
        }
        choice
    }
}

impl Default for PolicyRng {
    fn default() -> Self {
        Self::new()
    }
}
