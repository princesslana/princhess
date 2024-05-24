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

    let capacity = 16 * TrainingPosition::SIZE;
    let mut buffer = BufReader::with_capacity(capacity, file);

    let mut policy_inputs: [u64; state::POLICY_NUMBER_FEATURES] =
        [0; state::POLICY_NUMBER_FEATURES];
    let mut policy_outputs_from: [u64; 64] = [0; 64];
    let mut policy_outputs_to: [u64; 64] = [0; 64];

    let mut count = 0;

    while let Ok(buf) = buffer.fill_buf() {
        if buf.is_empty() {
            break;
        }

        let positions = TrainingPosition::read_batch(buf);

        for position in positions.iter() {
            let features = position.get_policy_features();
            let moves = position.moves();
            let state = State::from(position);

            for feature in features.iter() {
                policy_inputs[*feature] += 1;
            }

            for (m, _) in moves.iter() {
                let move_idx = state.move_to_index(*m);
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

    println!("policy inputs:");
    for (idx, input) in policy_inputs.iter().enumerate() {
        print!(
            "{:>8}/{:>5.2}%  ",
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
            "{:>8}/{:>5.2}%  ",
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
            "{:>8}/{:>5.2}%  ",
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
