use princhess::math::Rng;
use princhess::train::TrainingPosition;

use std::env;
use std::fs::File;
use std::io::BufWriter;
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

        for _ in 0..positions.len() * 8 {
            let i = rng.next_usize() % positions.len();
            let j = rng.next_usize() % positions.len();
            positions.swap(i, j);
        }

        println!("Done ({}ms).", start.elapsed().as_millis());

        let mut writer = BufWriter::new(File::create(format!("{}.shuffled", file)).unwrap());
        TrainingPosition::write_batch(&mut writer, positions).unwrap();
    }
}
