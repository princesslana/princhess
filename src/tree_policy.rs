use std::f32;

use crate::search::SCALE;
use crate::search_tree::{MoveInfoHandle, Moves};

pub fn choose_child(moves: Moves<'_>, cpuct: f32, is_root: bool) -> MoveInfoHandle<'_> {
    let total_visits = moves.map(MoveInfoHandle::visits).sum::<u64>() + 1;
    let sqrt_total_visits = (total_visits as f32).sqrt();
    let exploration_constant = (cpuct + cpuct * (total_visits / 8192) as f32) * SCALE;

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
        let policy_evaln = mov.policy();

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
