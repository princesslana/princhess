use princhess::math::Rng;
use princhess::train::TrainingPosition;

use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::time::Instant;

fn main() {
    let mut args = env::args();
    args.next();

    let files = args.collect::<Vec<String>>();

    let mut rng = Rng::default();

    for file in files {
        let mut bytes = std::fs::read(file.clone()).unwrap();
        let positions = TrainingPosition::read_batch_mut(&mut bytes);

        let start = Instant::now();
        println!("Shuffling {} positions from {}...", positions.len(), file);

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

        println!("\nDone ({}ms).", start.elapsed().as_millis());

        let mut writer = BufWriter::new(File::create(format!("{}.shuffled", file)).unwrap());
        TrainingPosition::write_batch(&mut writer, positions).unwrap();
    }
}
