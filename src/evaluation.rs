use chess::*;
use features::Model;
use mcts::{Evaluator, GameState};
use policy_features::evaluate_moves;
use search::{GooseMCTS, SCALE};
use shakmaty::Position;
use shakmaty_syzygy::Wdl;
use state::{MoveList, Player, State};
use tablebase::probe_tablebase_wdl;

const MATE_FACTOR: f32 = 1.1;

pub struct GooseEval {
    model: Model,
}

impl GooseEval {
    pub fn new(model: Model) -> Self {
        Self { model }
    }
}

impl Evaluator<GooseMCTS> for GooseEval {
    fn evaluate_new_state(&self, state: &State, moves: &MoveList) -> (Vec<f32>, i64, bool) {
        let features = state.features();
        let move_evaluations = evaluate_moves(state, &features, moves);
        let mut tb_hit = false;
        let state_evaluation = if moves.len() == 0 {
            let x = (MATE_FACTOR * SCALE) as i64;
            if state.shakmaty_board().is_check() {
                match state.current_player() {
                    chess::Color::White => -x,
                    chess::Color::Black => x,
                }
            } else {
                0
            }
        } else if let Some(wdl) = probe_tablebase_wdl(state.shakmaty_board()) {
            tb_hit = true;
            let win_score = SCALE as i64;
            match (wdl, state.board().side_to_move()) {
                (Wdl::Win, Color::White) => win_score,
                (Wdl::Loss, Color::White) => -win_score,
                (Wdl::Win, Color::Black) => -win_score,
                (Wdl::Loss, Color::Black) => win_score,
                _ => 0,
            }
        } else {
            (self.model.score(state, &features) * SCALE as f32) as i64
        };
        (move_evaluations, state_evaluation, tb_hit)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, player: &Player) -> i64 {
        match *player {
            Color::White => *evaln,
            Color::Black => -*evaln,
        }
    }
}
