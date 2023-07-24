use fastapprox::faster;
use std::f32;

use crate::search::SCALE;
use crate::search_tree::{MoveInfoHandle, Moves};

#[derive(Copy, Debug, Clone)]
pub struct PuctParameters {
    cpuct: f32,
    negamax_weight: f32,
}

impl PuctParameters {
    pub fn new(cpuct: f32, negamax_weight: f32) -> Self {
        Self {
            cpuct,
            negamax_weight,
        }
    }
}

pub fn puct(moves: Moves<'_>, parameters: PuctParameters, is_root: bool) -> MoveInfoHandle<'_> {
    let total_visits = moves.map(|v| u64::from(v.visits())).sum::<u64>() + 1;
    let sqrt_total_visits = (total_visits as f32).sqrt();

    let cpuct = parameters.cpuct;
    let negamax_weight = parameters.negamax_weight;

    let exploration_constant =
        (cpuct + cpuct * faster::ln(((total_visits + 8192) / 8192) as f32)) * SCALE;

    let explore_coef = exploration_constant * sqrt_total_visits;

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

        let negamax_evaln = mov.negamax() as f32 / (2. * SCALE) + 1.;
        let policy_evaln = if is_root {
            (1. - negamax_weight) * mov.policy() + negamax_weight * negamax_evaln * negamax_evaln
        } else {
            mov.policy()
        };

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
