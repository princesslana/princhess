use princhess::math::Rng;
use princhess::train::TrainingPosition;

use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let mut args = env::args();
    args.next();

    let files = args.collect::<Vec<String>>();

    let mut rng = Rng::default();

    let mut inputs = Vec::new();
    let mut total = 0;

    for path in files {
        let file = File::open(path).unwrap();
        let count = file.metadata().unwrap().len() as usize / TrainingPosition::SIZE;

        inputs.push((count, BufReader::new(file)));
        total += count;
    }

    let mut remaining = total;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut writer =
        BufWriter::new(File::create(format!("data/princhess-{timestamp}-all.data")).unwrap());

    println!(
        "Interleaving {} positions from {} files...",
        total,
        inputs.len()
    );

    while remaining > 0 {
        let mut choice = rng.next_usize() % remaining;
        let mut idx = 0;
        while inputs[idx].0 <= choice {
            choice -= inputs[idx].0;
            idx += 1;
        }

        let (count, reader) = &mut inputs[idx];
        let mut value = [0; TrainingPosition::SIZE];
        reader.read_exact(&mut value).unwrap();
        writer.write_all(&value).unwrap();

        remaining -= 1;
        *count -= 1;
        if *count == 0 {
            inputs.remove(idx);
        }

        if remaining % 65536 == 0 {
            let written = total - remaining;
            let pct_done = (written as f64 / total as f64) * 100.0;

            print!("Written {} positions ({:.2}%)\r", written, pct_done);
            io::stdout().flush().unwrap();
        }
    }

    println!("Done.");
}
