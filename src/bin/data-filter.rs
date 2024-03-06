use princhess::state::State;
use princhess::train::TrainingPosition;

use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::env;
use std::fs::File;
use std::time::Instant;

const BUFFER_COUNT: usize = 1 << 16;

fn main() {
    let mut args = env::args();
    args.next();

    let files = args.collect::<Vec<String>>();

    for input in files {
        let file = File::open(input.clone()).unwrap();
        let count = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

        let buffer_size = BUFFER_COUNT * TrainingPosition::SIZE;
        let mut buffer = BufReader::with_capacity(buffer_size, file);

        let mut writer = BufWriter::new(File::create(format!("{}.filtered", input)).unwrap());

        let start = Instant::now();
        println!("Filtering {} positions from {}...", count, input);

        let mut processed = 0;
        let mut filtered = 0;

        while let Ok(bytes) = buffer.fill_buf() {
            if bytes.is_empty() {
                break;
            }

            let data = TrainingPosition::read_batch(bytes);

            for position in data {
                let state = State::from(position);

                if state.is_check() {
                    filtered += 1;
                    continue;
                }

                TrainingPosition::write_batch(&mut writer, &[*position]).unwrap();
            }

            processed += data.len();

            print!(
                "{:>8} / {} ({:2}%) / {} ({:2}%)\r",
                processed,
                count,
                processed * 100 / count,
                filtered,
                filtered * 100 / processed
            );
            io::stdout().flush().unwrap();

            let consumed = bytes.len();
            buffer.consume(consumed);
        }

        println!("Filtered {} positions in {:?}", count, start.elapsed());
    }
}
