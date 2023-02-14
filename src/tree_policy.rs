use std::f32;

use crate::search_tree::{MoveInfoHandle, Moves, SearchHandle};
use crate::state::State;

#[derive(Clone, Debug)]
pub struct Puct {
    cpuct: f32,
}

impl Puct {
    pub fn new(cpuct: f32) -> Self {
        Self { cpuct }
    }

    pub fn choose_child<'a>(
        &self,
        _: &State,
        moves: Moves<'a>,
        handle: SearchHandle,
    ) -> MoveInfoHandle<'a> {
        let total_visits = moves.map(|x| x.visits()).sum::<u64>() + 1;
        let sqrt_total_visits = (total_visits as f32).sqrt();
        let exploration_constant = self.cpuct;

        let explore_coef = exploration_constant * sqrt_total_visits;

        let is_root = handle.is_root();

        let mut best_score = (f32::NEG_INFINITY, 1.);
        let mut choice = None;

        for mov in moves {
            if let Some(pc) = mov.get_move().promotion() {
                if !is_root && pc != shakmaty::Role::Queen {
                    continue;
                }
            }

            let sum_rewards = mov.sum_rewards() as f32;
            let child_visits = mov.visits();
            let policy_evaln = *mov.move_evaluation();

            let numerator = sum_rewards + explore_coef * policy_evaln;
            let denominator = (child_visits + 1) as f32;

            if choice.is_none() {
                choice = Some(mov);
                best_score = (numerator, denominator);
            } else {
                let a = numerator * best_score.1;
                let b = denominator * best_score.0;

                if a > b {
                    choice = Some(mov);
                    best_score = (numerator, denominator);
                }
            }
        }

        choice.unwrap()
    }

    pub fn validate_evaluations(&self, evalns: &[f32]) {
        for &x in evalns {
            assert!(x >= -1e-6, "Move evaluation is {x} (must be non-negative)");
        }
    }
}
