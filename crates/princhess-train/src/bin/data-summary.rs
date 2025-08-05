use arrayvec::ArrayVec;
use princhess::chess::{Color, File};
use princhess::nets::MoveIndex;
use princhess::state::{self, State, NUMBER_KING_BUCKETS, VALUE_NUMBER_FEATURES};
use std::env;
use std::fs::File as FsFile;
use std::io::{self, BufRead, BufReader, Write};

use princhess_train::analysis_utils::{king_bucket_name, piece_name, threat_bucket_name};
use princhess_train::data::TrainingPosition;
use princhess_train::policy::Phase;

fn main() {
    let mut args = env::args();
    args.next();

    let path = args.next().expect("no path given");

    let file = FsFile::open(path).expect("could not open file");
    let records = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

    let mut phase_win: [u64; 25] = [0; 25];
    let mut phase_draw: [u64; 25] = [0; 25];
    let mut phase_loss: [u64; 25] = [0; 25];

    let mut material_white: [u64; 25] = [0; 25];
    let mut material_draw: [u64; 25] = [0; 25];
    let mut material_black: [u64; 25] = [0; 25];

    let mut policy_inputs: [u64; state::POLICY_NUMBER_FEATURES] =
        [0; state::POLICY_NUMBER_FEATURES];

    let mut policy_outputs_sq: [u64; MoveIndex::SQ_COUNT] = [0; MoveIndex::SQ_COUNT];
    let mut policy_outputs_from_piece_sq: [u64; MoveIndex::FROM_PIECE_SQ_COUNT] =
        [0; MoveIndex::FROM_PIECE_SQ_COUNT];
    let mut policy_outputs_to_piece_sq: [u64; MoveIndex::TO_PIECE_SQ_COUNT] =
        [0; MoveIndex::TO_PIECE_SQ_COUNT];

    let mut middle_game_matched_count: u64 = 0;
    let mut endgame_matched_count: u64 = 0;

    // Pawn count analysis (0-16 pawns possible)
    let mut pawn_count_distribution: [u64; 17] = [0; 17];

    // King position analysis (using is_endgame and flip_stm logic)
    let mut mg_king_positions: [u64; 64] = [0; 64];
    let mut eg_king_positions: [u64; 64] = [0; 64];
    let mut value_endgame_count: u64 = 0;

    // Value feature analysis
    let mut value_inputs: [u64; VALUE_NUMBER_FEATURES * 2] = [0; VALUE_NUMBER_FEATURES * 2];
    let mut mg_king_buckets: [u64; NUMBER_KING_BUCKETS] = [0; NUMBER_KING_BUCKETS];
    let mut eg_king_buckets: [u64; NUMBER_KING_BUCKETS] = [0; NUMBER_KING_BUCKETS];

    // Piece-wise threat bucket analysis: [piece_type][threat_bucket]
    let mut mg_piece_threat_buckets: [[u64; 4]; 6] = [[0; 4]; 6];
    let mut eg_piece_threat_buckets: [[u64; 4]; 6] = [[0; 4]; 6];

    let mut count = 0;

    let mut first = true;

    while let Ok(buf) = buffer.fill_buf() {
        if buf.is_empty() {
            break;
        }

        let positions = TrainingPosition::read_buffer(buf);

        if first {
            first = false;
            println!("samples: {:?}", &positions[..10]);
        }

        for position in positions.iter() {
            let state = State::from(position);
            let moves = position.moves().iter().map(|(mv, _)| *mv).collect();

            let is_policy_mg = Phase::MiddleGame.matches(&state);
            let is_policy_eg = Phase::Endgame.matches(&state);
            let is_value_endgame = state.is_endgame();

            if is_policy_mg {
                middle_game_matched_count += 1;
            }
            if is_policy_eg {
                endgame_matched_count += 1;
            }
            if is_value_endgame {
                value_endgame_count += 1;
            }

            // Count total pawns on the board
            let pawn_count = state.board().pawns().count();
            pawn_count_distribution[pawn_count] += 1;

            // Track STM king positions with proper flipping (like value_features_map)
            let stm = state.side_to_move();
            let stm_ksq = state.board().king_of(stm);

            let flipped_king_sq = match (stm == Color::BLACK, stm_ksq.file() <= File::D) {
                (true, true) => stm_ksq.flip_rank().flip_file(),
                (true, false) => stm_ksq.flip_rank(),
                (false, true) => stm_ksq.flip_file(),
                (false, false) => stm_ksq,
            };

            if is_value_endgame {
                eg_king_positions[flipped_king_sq.index()] += 1;
            } else {
                mg_king_positions[flipped_king_sq.index()] += 1;
            }

            let mut features = ArrayVec::<usize, 64>::new();
            state.policy_features_map(|feature| features.push(feature));

            // Analyze value features and extract bucket distributions
            let mut value_features = ArrayVec::<usize, 64>::new();
            state.value_features_map(|feature| value_features.push(feature));

            for &feature in &value_features {
                value_inputs[feature] += 1;

                // Extract bucket information from feature index
                let is_nstm = feature >= VALUE_NUMBER_FEATURES;
                let base_feature = if is_nstm {
                    feature - VALUE_NUMBER_FEATURES
                } else {
                    feature
                };
                let bucket = base_feature / 768;
                let position = base_feature % 768;
                let king_bucket = bucket % NUMBER_KING_BUCKETS;
                let threat_bucket = bucket / NUMBER_KING_BUCKETS;

                // Extract piece type from position
                let side_offset = if position >= 384 { 384 } else { 0 };
                let piece_position = position - side_offset;
                let piece_idx = piece_position / 64;

                if is_value_endgame {
                    eg_king_buckets[king_bucket] += 1;
                    eg_piece_threat_buckets[piece_idx][threat_bucket] += 1;
                } else {
                    mg_king_buckets[king_bucket] += 1;
                    mg_piece_threat_buckets[piece_idx][threat_bucket] += 1;
                }
            }

            let material_balance_idx = (state.material_balance() + 12).clamp(0, 24) as usize;

            match position.stm_relative_result() {
                1 => phase_win[state.phase()] += 1,
                0 => phase_draw[state.phase()] += 1,
                -1 => phase_loss[state.phase()] += 1,
                _ => (),
            }

            match position.white_relative_result() {
                1 => material_white[material_balance_idx] += 1,
                0 => material_draw[material_balance_idx] += 1,
                -1 => material_black[material_balance_idx] += 1,
                _ => (),
            }

            for feature in features.iter() {
                policy_inputs[*feature] += 1;
            }

            for move_idx in state.moves_to_indexes(&moves) {
                policy_outputs_sq[move_idx.from_sq()] += 1;
                policy_outputs_sq[move_idx.to_sq()] += 1;
                policy_outputs_from_piece_sq[move_idx.from_piece_sq_index()] += 1;
                policy_outputs_to_piece_sq[move_idx.to_piece_sq_index()] += 1;
            }
        }

        count += positions.len();

        print!(
            "{:>8} / {} ({:2}%) \r",
            count,
            records,
            count * 100 / records
        );
        io::stdout().flush().unwrap();

        let consumed = buf.len();
        buffer.consume(consumed);
    }

    println!("records: {records}");

    println!("phase:");
    for idx in 0..25 {
        let (w, d, l) = (phase_win[idx], phase_draw[idx], phase_loss[idx]);
        let total = w + d + l;

        println!(
            "{:>2}: {:>15}/{:>5.2}%  +{:>2} ={:>2} -{:>2} %",
            idx,
            total,
            total as f32 / records as f32 * 100.0,
            w * 100 / total,
            d * 100 / total,
            l * 100 / total
        );
    }

    println!("material:");
    for idx in 0..25 {
        let (w, d, l) = (material_white[idx], material_draw[idx], material_black[idx]);
        let total = w + d + l;

        println!(
            "{:>2}: {:>15}/{:>5.2}%  +{:>2} ={:>2} -{:>2} %",
            (idx as i64) - 12,
            total,
            total as f32 / records as f32 * 100.0,
            w * 100 / total,
            d * 100 / total,
            l * 100 / total
        );
    }

    println!("pawn count:");
    for (pawn_count, &count) in pawn_count_distribution.iter().enumerate() {
        if count > 0 {
            println!(
                "{:>2}: {:>15}/{:>5.2}%",
                pawn_count,
                count,
                count as f32 / records as f32 * 100.0
            );
        }
    }

    println!("policy inputs:");
    for (idx, input) in policy_inputs.iter().enumerate() {
        print!(
            "{:>9}/{:>5.2}%  ",
            input,
            *input as f32 / records as f32 * 100.0
        );

        if idx % 8 == 7 {
            println!();
        }
        if idx % 64 == 63 {
            println!();
        }
    }

    println!("policy outputs (sq):");
    for (idx, output) in policy_outputs_sq.iter().enumerate() {
        print!(
            "{:>9}/{:>5.2}%  ",
            output,
            *output as f32 / records as f32 * 100.0
        );

        if idx % 8 == 7 {
            println!();
        }
        if idx % 64 == 63 {
            println!();
        }
    }

    println!("policy outputs (from piece sq):");
    for (idx, output) in policy_outputs_from_piece_sq.iter().enumerate() {
        print!(
            "{:>9}/{:>5.2}%  ",
            output,
            *output as f32 / records as f32 * 100.0
        );

        if idx % 8 == 7 {
            println!();
        }
        if idx % 64 == 63 {
            println!();
        }
    }

    println!("policy outputs (to piece sq):");
    for (idx, output) in policy_outputs_to_piece_sq.iter().enumerate() {
        print!(
            "{:>9}/{:>5.2}%  ",
            output,
            *output as f32 / records as f32 * 100.0
        );

        if idx % 8 == 7 {
            println!();
        }
        if idx % 64 == 63 {
            println!();
        }
    }

    let value_middlegame_count = records as u64 - value_endgame_count;

    println!("\nPositions by training phase (Policy Network):");
    println!(
        "  Middle Game: {:>15} ({:>5.2}%)",
        middle_game_matched_count,
        middle_game_matched_count as f32 / records as f32 * 100.0
    );
    println!(
        "  Endgame:     {:>15} ({:>5.2}%)",
        endgame_matched_count,
        endgame_matched_count as f32 / records as f32 * 100.0
    );

    println!("\nPositions by game phase (Value Network):");
    println!(
        "  Middle Game: {:>15} ({:>5.2}%)",
        value_middlegame_count,
        value_middlegame_count as f32 / records as f32 * 100.0
    );
    println!(
        "  Endgame:     {:>15} ({:>5.2}%)",
        value_endgame_count,
        value_endgame_count as f32 / records as f32 * 100.0
    );

    println!("\nSTM King positions in Value Middle Game (after flipping):");
    println!("   a    b    c    d    e    f    g    h");
    for rank in (0..8).rev() {
        print!("{} ", rank + 1);
        for file in 0..8 {
            let sq = rank * 8 + file;
            let count = mg_king_positions[sq];
            if count > 0 {
                print!(
                    "{:>4.1}% ",
                    count as f32 / value_middlegame_count as f32 * 100.0
                );
            } else {
                print!("  .   ");
            }
        }
        println!();
    }

    println!("\nSTM King positions in Value Endgame (after flipping):");
    println!("   a    b    c    d    e    f    g    h");
    for rank in (0..8).rev() {
        print!("{} ", rank + 1);
        for file in 0..8 {
            let sq = rank * 8 + file;
            let count = eg_king_positions[sq];
            if count > 0 {
                print!(
                    "{:>4.1}% ",
                    count as f32 / value_endgame_count as f32 * 100.0
                );
            } else {
                print!("  .   ");
            }
        }
        println!();
    }

    // Value feature analysis output
    println!("\n=== VALUE NETWORK FEATURE ANALYSIS ===");

    println!("\nKing bucket distribution (from value features):");
    let total_mg_king_features: u64 = mg_king_buckets.iter().sum();
    let total_eg_king_features: u64 = eg_king_buckets.iter().sum();

    println!("Middle Game:");
    for (bucket, &count) in mg_king_buckets.iter().enumerate() {
        let bucket_name = king_bucket_name(bucket);
        let percentage = if total_mg_king_features > 0 {
            count as f32 / total_mg_king_features as f32 * 100.0
        } else {
            0.0
        };
        println!("  King bucket {bucket} ({bucket_name}): {count:>10} ({percentage:>5.1}%)");
    }

    println!("Endgame:");
    for (bucket, &count) in eg_king_buckets.iter().enumerate() {
        let bucket_name = king_bucket_name(bucket);
        let percentage = if total_eg_king_features > 0 {
            count as f32 / total_eg_king_features as f32 * 100.0
        } else {
            0.0
        };
        println!("  King bucket {bucket} ({bucket_name}): {count:>10} ({percentage:>5.1}%)");
    }

    println!("\nThreat bucket distribution (from value features):");

    // Calculate overall totals by summing across all pieces
    let mut mg_threat_totals = [0u64; 4];
    let mut eg_threat_totals = [0u64; 4];

    for piece_idx in 0..6 {
        for threat_bucket in 0..4 {
            mg_threat_totals[threat_bucket] += mg_piece_threat_buckets[piece_idx][threat_bucket];
            eg_threat_totals[threat_bucket] += eg_piece_threat_buckets[piece_idx][threat_bucket];
        }
    }

    let total_mg_threat_features: u64 = mg_threat_totals.iter().sum();
    let total_eg_threat_features: u64 = eg_threat_totals.iter().sum();

    println!("Middle Game:");
    for (bucket, &count) in mg_threat_totals.iter().enumerate() {
        let bucket_name = threat_bucket_name(bucket);
        let percentage = if total_mg_threat_features > 0 {
            count as f32 / total_mg_threat_features as f32 * 100.0
        } else {
            0.0
        };
        println!("  Threat bucket {bucket} ({bucket_name}): {count:>10} ({percentage:>5.1}%)");
    }

    println!("Endgame:");
    for (bucket, &count) in eg_threat_totals.iter().enumerate() {
        let bucket_name = threat_bucket_name(bucket);
        let percentage = if total_eg_threat_features > 0 {
            count as f32 / total_eg_threat_features as f32 * 100.0
        } else {
            0.0
        };
        println!("  Threat bucket {bucket} ({bucket_name}): {count:>10} ({percentage:>5.1}%)");
    }

    println!("\nThreat bucket distribution by piece type:");

    println!("Middle Game:");
    for (piece_idx, piece_buckets) in mg_piece_threat_buckets.iter().enumerate() {
        let piece_total: u64 = piece_buckets.iter().sum();
        if piece_total > 0 {
            println!("  {}:", piece_name(piece_idx).to_uppercase());
            for (bucket, &count) in piece_buckets.iter().enumerate() {
                let bucket_name = threat_bucket_name(bucket);
                let percentage = count as f32 / piece_total as f32 * 100.0;
                println!("    {bucket_name}: {count:>8} ({percentage:>5.1}%)");
            }
        }
    }

    println!("Endgame:");
    for (piece_idx, piece_buckets) in eg_piece_threat_buckets.iter().enumerate() {
        let piece_total: u64 = piece_buckets.iter().sum();
        if piece_total > 0 {
            println!("  {}:", piece_name(piece_idx).to_uppercase());
            for (bucket, &count) in piece_buckets.iter().enumerate() {
                let bucket_name = threat_bucket_name(bucket);
                let percentage = count as f32 / piece_total as f32 * 100.0;
                println!("    {bucket_name}: {count:>8} ({percentage:>5.1}%)");
            }
        }
    }
}
