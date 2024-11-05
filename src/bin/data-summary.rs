use princhess::policy::MoveIndex;
use princhess::state::{self, State};
use princhess::train::TrainingPosition;

use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

fn main() {
    let mut args = env::args();
    args.next();

    let path = args.next().expect("no path given");

    let file = File::open(path).expect("could not open file");
    let records = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

    let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

    let mut phase_win: [u64; 25] = [0; 25];
    let mut phase_draw: [u64; 25] = [0; 25];
    let mut phase_loss: [u64; 25] = [0; 25];

    let mut policy_inputs: [u64; state::POLICY_NUMBER_FEATURES] =
        [0; state::POLICY_NUMBER_FEATURES];
    let mut policy_outputs_from: [u64; MoveIndex::FROM_COUNT] = [0; MoveIndex::FROM_COUNT];
    let mut policy_outputs_to: [u64; MoveIndex::TO_COUNT] = [0; MoveIndex::TO_COUNT];

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
            let features = position.get_policy_features();
            let moves = position.moves().iter().map(|(mv, _)| *mv).collect();
            let state = State::from(position);

            match position.stm_relative_result() {
                1 => phase_win[state.phase()] += 1,
                0 => phase_draw[state.phase()] += 1,
                -1 => phase_loss[state.phase()] += 1,
                _ => (),
            }

            for feature in features.iter() {
                policy_inputs[*feature] += 1;
            }

            for move_idx in state.moves_to_indexes(&moves) {
                policy_outputs_from[move_idx.from_index()] += 1;
                policy_outputs_to[move_idx.to_index()] += 1;
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

    println!("records: {}", records);

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

    println!("policy outputs (from):");
    for (idx, output) in policy_outputs_from.iter().enumerate() {
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

    println!("policy outputs (to):");
    for (idx, output) in policy_outputs_to.iter().enumerate() {
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
}
