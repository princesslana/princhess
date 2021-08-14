extern crate rand;
use self::rand::{XorShiftRng, Rng, SeedableRng};

use mcts::MCTS;

use std;
use search_tree::*;

pub struct Fraction(pub f32, pub f32);

impl From<f32> for Fraction {
    fn from(x: f32) -> Fraction {
        Fraction(x, 1.)
    }
}

pub trait TreePolicy<Spec: MCTS<TreePolicy=Self>>: Sync + Sized {
    type MoveEvaluation: Sync + Send;
    type ThreadLocalData: Default;

    fn choose_child<'a>(&self, state: &Spec::State, moves: Moves<'a, Spec>, handle: SearchHandle<Spec>)
        -> MoveInfoHandle<'a, Spec>;
    fn validate_evaluations(&self, _evalns: &[Self::MoveEvaluation]) {}
    fn reset(self) -> Self { self } // TODO put this on everything
}

#[derive(Clone, Debug)]
pub struct UCTPolicy {
    exploration_constant: f32,
}

impl UCTPolicy {
    pub fn new(exploration_constant: f32) -> Self {
        assert!(exploration_constant > 0.0,
            "exploration constant is {} (must be positive)",
            exploration_constant);
        Self {exploration_constant}
    }

    pub fn exploration_constant(&self) -> f32 {
        self.exploration_constant
    }
}

#[derive(Clone, Debug)]
pub struct AlphaGoPolicy {
    exploration_constant: f32,
}

impl AlphaGoPolicy {
    pub fn new(exploration_constant: f32) -> Self {
        Self {exploration_constant}
    }
    pub fn exploration_constant(&self) -> f32 {
        self.exploration_constant
    }
}

impl<Spec: MCTS<TreePolicy=Self>> TreePolicy<Spec> for UCTPolicy
{
    type ThreadLocalData = PolicyRng;
    type MoveEvaluation = ();

    fn choose_child<'a>(&self, _: &Spec::State, moves: Moves<'a, Spec>, mut handle: SearchHandle<Spec>)
        -> MoveInfoHandle<'a, Spec>
    {
        let total_visits = moves.map(|x| x.visits()).sum::<u64>();
        let adjusted_total = (total_visits + 1) as f32;
        let ln_adjusted_total = adjusted_total.ln();
        handle.thread_data().policy_data.select_by_key(moves, |mov| {
            let sum_rewards = mov.sum_rewards();
            let child_visits = mov.visits();
            // http://mcts.ai/pubs/mcts-survey-master.pdf
            let explore_term = if child_visits == 0 {
                std::f32::INFINITY
            } else {
                2.0 * (ln_adjusted_total / child_visits as f32).sqrt()
            };
            let mean_action_value = sum_rewards as f32 / adjusted_total;
            (self.exploration_constant * explore_term + mean_action_value).into()
        }).unwrap()
    }
}

impl<Spec: MCTS<TreePolicy=Self>> TreePolicy<Spec> for AlphaGoPolicy
{
    type ThreadLocalData = PolicyRng;
    type MoveEvaluation = f32;

    fn choose_child<'a>(&self, _: &Spec::State, moves: Moves<'a, Spec>, mut handle: SearchHandle<Spec>)
        -> MoveInfoHandle<'a, Spec>
    {
        let total_visits = moves.map(|x| x.visits()).sum::<u64>() + 1;
        let sqrt_total_visits = (total_visits as f32).sqrt();
        let explore_coef = self.exploration_constant * sqrt_total_visits;
        handle.thread_data().policy_data.select_by_key(moves, |mov| {
            let sum_rewards = mov.sum_rewards() as f32;
            let child_visits = mov.visits();
            let policy_evaln = *mov.move_evaluation() as f32;
            Fraction(
                sum_rewards + explore_coef * policy_evaln,
                (child_visits + 1) as f32)
        }).unwrap()
    }

    fn validate_evaluations(&self, evalns: &[f32]) {
        for &x in evalns {
            assert!(x >= -1e-6,
                "Move evaluation is {} (must be non-negative)",
                x);
        }
        if evalns.len() >= 1 {
            let evaln_sum: f32 = evalns.iter().sum();
            assert!((evaln_sum - 1.0).abs() < 0.1,
                "Sum of evaluations is {} (should sum to 1)",
                evaln_sum);
        }
    }
}

#[derive(Clone)]
pub struct PolicyRng {
    pub rng: XorShiftRng
}

impl PolicyRng {
    pub fn new() -> Self {
        let rng = SeedableRng::from_seed([1, 2, 3, 4]);
        Self {rng}
    }

    pub fn select_by_key<T, Iter, KeyFn>(&mut self, elts: Iter, mut key_fn: KeyFn) -> Option<T>
        where Iter: Iterator<Item=T>, KeyFn: FnMut(&T) -> Fraction
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
            } else if a == b {
                num_optimal += 1;
                if self.rng.gen_weighted_bool(num_optimal) {
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
