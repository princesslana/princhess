use chess::*;
use features::Model;
use mcts::Evaluator;
use policy_features::evaluate_moves;
use search::{GooseMCTS, SCALE};
use state::{MoveList, Outcome, Player, State};

pub struct GooseEval {
    model: Model,
}

impl GooseEval {
    pub fn new(model: Model) -> Self {
        Self { model }
    }
}

impl Evaluator<GooseMCTS> for GooseEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(&self, state: &State, moves: &MoveList) -> (Vec<f32>, i64) {
        let move_evaluations = evaluate_moves(state, moves.as_slice());
        let state_evaluation = if moves.len() == 0 {
            let x = SCALE as i64;
            match state.outcome() {
                Outcome::Draw => 0,
                Outcome::WhiteWin => x,
                Outcome::BlackWin => -x,
                Outcome::Ongoing => unreachable!(),
            }
        } else {
            (self.model.score(state) * SCALE as f32) as i64
        };
        (move_evaluations, state_evaluation)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, player: &Player) -> i64 {
        match *player {
            Color::White => *evaln,
            Color::Black => -*evaln,
        }
    }
}
