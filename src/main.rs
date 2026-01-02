use std::env;

use princhess::uci::Uci;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut uci_instance = Uci::new();

    if args.len() > 1 {
        for command in args.iter().skip(1) {
            let (_, _) = uci_instance.handle_command(command, false);
        }
    } else {
        uci_instance.main_loop();
    }
}
