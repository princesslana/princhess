use fastapprox::faster;
use std::f32;

use crate::search::SCALE;
use crate::search_tree::MoveEdge;

pub fn choose_child(moves: &[MoveEdge], cpuct: f32, cpuct_tau: f32, fpu: i64) -> &MoveEdge {
    let total_visits = moves.iter().map(|v| u64::from(v.visits())).sum::<u64>() + 1;

    let explore_coef =
        (cpuct * faster::exp(cpuct_tau * faster::ln(total_visits as f32)) * SCALE) as i64;

    let mut best_move = &moves[0];
    let mut best_score = i64::MIN;

    for mov in moves {
        let sum_rewards = mov.sum_rewards();
        let child_visits = i64::from(mov.visits());
        let policy_evaln = mov.policy();

        let q = if child_visits > 0 {
            sum_rewards / child_visits
        } else {
            fpu
        };

        let u = explore_coef * i64::from(policy_evaln) / ((child_visits + 1) * SCALE as i64);

        let score = q + u;

        if score > best_score {
            best_score = score;
            best_move = mov;
        }
    }

    best_move
}
