use princhess_training::train::TrainingPosition;

use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

fn main() {
    let mut args = env::args();
    args.next();

    let truncate_to = args.next().unwrap().parse::<usize>().unwrap();
    let files = args.collect::<Vec<String>>();

    for input in files {
        let file = File::open(input.clone()).unwrap();
        let positions = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

        let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

        let out_file = format!("{}.{}m.truncated", input, truncate_to);
        let mut writer = BufWriter::new(File::create(out_file).unwrap());

        let start = Instant::now();
        println!(
            "Truncating {} positions from {} to {}m positions...",
            positions, input, truncate_to
        );

        let mut processed = 0;

        while let Ok(bytes) = buffer.fill_buf() {
            if bytes.is_empty() {
                break;
            }

            let data = TrainingPosition::read_buffer(bytes);
            TrainingPosition::write_buffer(&mut writer, data);

            processed += data.len();

            print!(
                "{:>8} / {} ({:2}%)\r",
                processed,
                truncate_to * 1000000,
                processed * 100 / (truncate_to * 1000000),
            );
            io::stdout().flush().unwrap();

            let consumed = bytes.len();
            buffer.consume(consumed);

            if processed >= truncate_to * 1000000 {
                break;
            }
        }

        println!("Wrote {}m positions in {:?}", truncate_to, start.elapsed());
    }
}
