use mcts::{Evaluator, SearchHandle};
use search::{GooseMCTS, SCALE};
use state::{State, Player, MoveList};
use features::Model;
use policy_features::evaluate_moves;
use chess::*;

pub struct GooseEval {
    model: Model
}

impl Evaluator<GooseMCTS> for GooseEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(&self, state: &State, moves: &MoveList,
                          _: Option<SearchHandle<GooseMCTS>>) -> (Vec<f32>, i64) {
        let move_evaluations = evaluate_moves(state, moves.as_slice());
        let state_evaluation = if moves.len() == 0 {
            let x = SCALE as i64;
            match state.outcome() {
                BoardStatus::Stalemate => 0,
                BoardStatus::Checkmate =>
                    if state.board().side_to_move() == Color::White {-x} else {x},
                BoardStatus::Ongoing => unreachable!(),
            }
        } else {
            (self.model.score(state, moves.as_slice()) * SCALE as f32) as i64
        };
        (move_evaluations, state_evaluation)
    }
    fn evaluate_existing_state(&self, _: &State, evaln: &i64,
                               _: SearchHandle<GooseMCTS>) -> i64 {
        *evaln
    }
    fn interpret_evaluation_for_player(&self, evaln: &i64, player: &Player) -> i64 {
        match *player {
            Color::White => *evaln,
            Color::Black => -*evaln,
        }
    }
}

impl From<Model> for GooseEval {
    fn from(m: Model) -> Self {
        Self {model: m}
    }
}

#[cfg(test)]
mod tests {
    use search::Search;
    use mcts::GameState;
    use super::*;
    use float_ord::FloatOrd;

    fn assert_find_move(fen: &str, desired: &str) -> Vec<State> {
        let pv_len = 15;
        let state = State::from_fen(fen).unwrap();
        let moves = state.available_moves();
        let moves = moves.as_slice();
        let evalns = evaluate_moves(&state, &moves);
        let mut paired: Vec<_> = moves.iter().zip(evalns.iter()).collect();
        paired.sort_by_key(|x| FloatOrd(*x.1));
        for (a, b) in paired {
            println!("policy: {} {}", a, b);
        }
        let mut manager = Search::create_manager(state);
        // for _ in 0..5 {
        manager.playout_n(1_000_000);
        println!("\n\nMOVES");
        manager.tree().display_moves();
        // }
        println!("Principal variation");
        let mov = manager.best_move().unwrap();
        for state in manager.principal_variation_states(pv_len) {
            println!("{}", state.board());
        }
        for info in manager.principal_variation_info(pv_len) {
            println!("{}", info);
        }
        println!("{}", manager.tree().diagnose());
        assert!(format!("{}", mov).starts_with(desired),
                "expected {}, got {}",
                desired,
                mov);
        manager.principal_variation_states(pv_len)
    }

    #[test]
    fn mate_in_one() {
        assert_find_move("6k1/8/6K1/8/8/8/8/R7 w - - 0 0", "a1a8");;
    }

    #[test]
    fn mate_in_six() {
        assert_find_move("5q2/6Pk/8/6K1/8/8/8/8 w - - 0 0", "g7f8r");
    }

    #[test]
    #[ignore]
    fn take_the_bishop() {
        assert_find_move("r3k2r/ppp1q1pp/2n1b3/8/3p4/6p1/PPPNQPP1/2K1RB1R w kq - 0 16", "Re1xe6");
    }

    #[test]
    #[ignore]
    fn what_happened() {
        assert_find_move("2k1r3/ppp2pp1/2nb1n1p/1q1rp3/8/2QPBNPP/PP2PPBK/2RR4 b - - 9 20", "foo");
    }

    #[test]
    #[ignore]
    fn what_happened_2() {
        assert_find_move("2r4r/ppB3p1/2n2k1p/1N5q/1b3Qn1/6Pb/PP2PPBP/R4RK1 b - - 10 18", "foo");
    }

    #[test]
    #[ignore]
    fn checkmating() {
        let states = assert_find_move("8/8/8/3k4/1Q6/K7/8/8 w - - 8 59", "");
        assert!(states[states.len() - 1].outcome() == BoardStatus::Checkmate);
    }

    #[test]
    #[ignore]
    fn interesting() {
        assert_find_move("2kr4/pp2bp1p/3p4/5b1Q/4q1r1/N4P2/PPPP2PP/R1B2RK1 b - -", "?");
    }
}
