use fastapprox::faster;
use std::f32;

use crate::graph::MoveEdge;
use crate::options::MctsOptions;
use crate::search::SCALE;

pub fn choose_child<'a>(moves: &'a [MoveEdge], fpu: i32, options: &MctsOptions) -> &'a MoveEdge {
    let total_visits = moves.iter().map(|v| u64::from(v.visits())).sum::<u64>() + 1;

    let explore_coef = (options.cpuct
        * faster::exp(options.cpuct_tau * faster::ln(total_visits as f32))
        * SCALE) as i64;

    let mut best_move = &moves[0];
    let mut best_score = i64::MIN;

    for mov in moves {
        let reward = mov.reward();
        let policy_evaln = mov.policy();

        let q = i64::from(if reward.visits > 0 {
            reward.average
        } else {
            fpu
        });

        let u = explore_coef * i64::from(policy_evaln)
            / ((i64::from(reward.visits) + 1) * SCALE as i64);

        let score = q + u;

        if score > best_score {
            best_score = score;
            best_move = mov;
        }
    }

    best_move
}
