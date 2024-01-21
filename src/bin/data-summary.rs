use princhess::train::TrainingPosition;

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let mut args = env::args();
    args.next();

    let path = args.next().expect("no path given");

    let file = File::open(path).expect("could not open file");
    let records = file.metadata().unwrap().len() / TrainingPosition::SIZE as u64;

    let capacity = 16 * TrainingPosition::SIZE;
    let mut buffer = BufReader::with_capacity(capacity, file);

    println!("records: {}", records);

    if let Ok(buf) = buffer.fill_buf() {
        let positions = TrainingPosition::read_batch(buf);

        println!("first 8 positions:");
        for p in positions.iter().take(8) {
            println!("{:?}", p);
        }
    }
}
