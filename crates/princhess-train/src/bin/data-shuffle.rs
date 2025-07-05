use bytemuck::allocation;
use princhess::math::Rng;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use princhess_train::data::TrainingPosition;

fn main() {
    let mut args = env::args();
    args.next();

    let files = args.collect::<Vec<String>>();

    let mut rng = Rng::default();

    for input in files {
        let file = File::open(input.clone()).unwrap();
        let mut positions =
            Vec::with_capacity(file.metadata().unwrap().len() as usize / TrainingPosition::SIZE);

        let mut buffer = BufReader::with_capacity(TrainingPosition::BUFFER_SIZE, file);

        let start = Instant::now();

        println!("Loading {input}...");

        while let Ok(bytes) = buffer.fill_buf() {
            if bytes.is_empty() {
                break;
            }

            let data = TrainingPosition::read_buffer(bytes);
            positions.extend_from_slice(data);

            let consumed = bytes.len();
            buffer.consume(consumed);
        }

        println!("Shuffling {} positions from {}...", positions.len(), input);

        for i in 0..positions.len() - 1 {
            let j = rng.next_usize() % (positions.len() - i);
            positions.swap(i, i + j);
            if i % 16 == 0 {
                print!(
                    "{:>8} / {} ({:2}%)\r",
                    i,
                    positions.len(),
                    i * 100 / positions.len()
                );
                io::stdout().flush().unwrap();
            }
        }

        let mut output_path = PathBuf::from(&input);
        output_path.set_extension("shuffled");
        let output_filename = output_path.to_string_lossy();

        println!("Saving {output_filename}...");

        let mut writer = BufWriter::new(File::create(&output_path).unwrap());
        let mut buffer: Box<[TrainingPosition; TrainingPosition::BUFFER_COUNT]> =
            allocation::zeroed_box();

        while !positions.is_empty() {
            let chunk_size = positions.len().min(TrainingPosition::BUFFER_COUNT);
            buffer[..chunk_size]
                .copy_from_slice(&positions.drain(..chunk_size).collect::<Vec<_>>());
            TrainingPosition::write_buffer(&mut writer, &buffer[..chunk_size]);
        }

        println!("Done ({}ms).", start.elapsed().as_millis());
    }
}
