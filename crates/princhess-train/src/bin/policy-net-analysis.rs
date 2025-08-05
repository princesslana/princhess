use princhess::chess::Square;
use princhess::nets::Accumulator;
use princhess::quantized_policy::{QuantizedPolicyNetwork, ATTENTION_SIZE, INPUT_SIZE};
use std::env;
use std::fs;

const DEAD_FEATURE_THRESHOLD: i32 = 100;
const SIGNIFICANT_WEIGHT_THRESHOLD: i16 = 5;
const ATTENTION_ACTIVE_THRESHOLD: i16 = 10;
const LOW_UTILIZATION_THRESHOLD: f64 = 0.1;

fn main() {
    let mut args = env::args();
    args.next();

    let network_path = args
        .next()
        .expect("Usage: network-analysis <path-to-policy.bin>");

    println!("Analyzing policy network: {network_path}");
    println!();

    let network_bytes = fs::read(&network_path)
        .unwrap_or_else(|e| panic!("Failed to read network file {network_path}: {e}"));

    let network: &QuantizedPolicyNetwork = bytemuck::try_from_bytes(&network_bytes)
        .unwrap_or_else(|e| panic!("Failed to parse network file as QuantizedPolicyNetwork: {e}"));

    analyze_network_weights(network);
}

fn analyze_network_weights(network: &QuantizedPolicyNetwork) {
    println!("=== WEIGHT ANALYSIS ===");

    println!("\n--- Square Subnet Analysis ---");
    let zero_weight_squares = analyze_subnets(network, true);

    println!("\n--- Piece-Square Subnet Analysis ---");
    let zero_weight_piece_sqs = analyze_subnets(network, false);

    println!("\n--- Overall Weight Statistics ---");
    let all_sq_weights = collect_all_weights(
        network.sq_count(),
        |idx| network.sq_subnet_bias(idx),
        |idx, feat_idx| network.sq_subnet_weight(idx, feat_idx),
    );

    let all_piece_sq_weights = collect_all_weights(
        network.piece_sq_count(),
        |idx| network.piece_sq_subnet_bias(idx),
        |idx, feat_idx| network.piece_sq_subnet_weight(idx, feat_idx),
    );

    println!("Square subnet weights:");
    print_weight_stats(&all_sq_weights);

    println!("Piece-square subnet weights:");
    print_weight_stats(&all_piece_sq_weights);

    analyze_attention_utilization(network);

    println!("\n--- Network Utilization Summary ---");
    let total_sq_subnets = network.sq_count();
    let unused_sq_subnets = zero_weight_squares.len();
    let sq_utilization =
        100.0 * (total_sq_subnets - unused_sq_subnets) as f64 / total_sq_subnets as f64;

    let total_piece_sq_subnets = network.piece_sq_count();
    let unused_piece_sq_subnets = zero_weight_piece_sqs.len();
    let piece_sq_utilization = 100.0 * (total_piece_sq_subnets - unused_piece_sq_subnets) as f64
        / total_piece_sq_subnets as f64;

    println!(
        "Square subnet utilization: {:.1}% ({}/{})",
        sq_utilization,
        total_sq_subnets - unused_sq_subnets,
        total_sq_subnets
    );
    println!(
        "Piece-square subnet utilization: {:.1}% ({}/{})",
        piece_sq_utilization,
        total_piece_sq_subnets - unused_piece_sq_subnets,
        total_piece_sq_subnets
    );

    if sq_utilization < 80.0 || piece_sq_utilization < 80.0 {
        println!("⚠️  Low subnet utilization detected - network may be over-parameterized");
    }

    if zero_weight_squares.len() > 10 || zero_weight_piece_sqs.len() > 50 {
        println!("⚠️  Many unused features detected - consider architecture simplification");
    }
}

fn analyze_subnet_weights<GetBias, GetWeight, NameFn>(
    network: &QuantizedPolicyNetwork,
    count: usize,
    subnet_type: &str,
    get_bias: GetBias,
    get_weight: GetWeight,
    name_fn: NameFn,
) -> Vec<usize>
where
    GetBias: Fn(&QuantizedPolicyNetwork, usize) -> Accumulator<i16, ATTENTION_SIZE>,
    GetWeight: Fn(&QuantizedPolicyNetwork, usize, usize) -> &Accumulator<i16, ATTENTION_SIZE>,
    NameFn: Fn(&QuantizedPolicyNetwork, usize) -> (String, String),
{
    let mut weight_sums = Vec::new();
    let mut zero_weights = Vec::new();

    for idx in 0..count {
        let bias = get_bias(network, idx);
        let weight_sum: i32 = (0..INPUT_SIZE)
            .map(|feat_idx| {
                let weights = get_weight(network, idx, feat_idx);
                weights.vals.iter().map(|&w| w.abs() as i32).sum::<i32>()
            })
            .sum();
        let bias_sum: i32 = bias.vals.iter().map(|&b| b.abs() as i32).sum();
        let total_magnitude = weight_sum + bias_sum;

        weight_sums.push((idx, total_magnitude));

        if total_magnitude < DEAD_FEATURE_THRESHOLD {
            zero_weights.push(idx);
        }
    }

    weight_sums.sort_by_key(|(_, weight)| std::cmp::Reverse(*weight));

    println!("Top 10 most important {subnet_type}:");
    for (i, (idx, weight)) in weight_sums.iter().take(10).enumerate() {
        let (name, extra) = name_fn(network, *idx);
        println!(
            "  {}: {}{} (weight magnitude: {})",
            i + 1,
            name,
            extra,
            weight
        );
    }

    println!("\nBottom 10 least important {subnet_type}:");
    for (i, (idx, weight)) in weight_sums.iter().rev().take(10).enumerate() {
        let (name, extra) = name_fn(network, *idx);
        println!(
            "  {}: {}{} (weight magnitude: {})",
            i + 1,
            name,
            extra,
            weight
        );
    }

    if !zero_weights.is_empty() {
        println!(
            "\nPotentially unused {subnet_type} ({} total):",
            zero_weights.len()
        );
        for idx in zero_weights.iter().take(10) {
            let (name, _) = name_fn(network, *idx);
            println!("  {name}");
        }
        if zero_weights.len() > 10 {
            println!("  ... and {} more", zero_weights.len() - 10);
        }
    }

    zero_weights
}

fn analyze_subnets(network: &QuantizedPolicyNetwork, is_square: bool) -> Vec<usize> {
    if is_square {
        analyze_subnet_weights(
            network,
            network.sq_count(),
            "squares",
            |net, idx| net.sq_subnet_bias(idx),
            |net, idx, feat_idx| net.sq_subnet_weight(idx, feat_idx),
            |net, idx| {
                let name = square_index_to_name(idx);
                let avg_sign = calculate_subnet_avg_sign(net, idx, true);
                let sign_str = format_sign(avg_sign);
                (format!("{name}{sign_str}"), String::new())
            },
        )
    } else {
        analyze_subnet_weights(
            network,
            network.piece_sq_count(),
            "piece-square combinations",
            |net, idx| net.piece_sq_subnet_bias(idx),
            |net, idx, feat_idx| net.piece_sq_subnet_weight(idx, feat_idx),
            |net, idx| {
                let name = decode_piece_sq_index(idx);
                let avg_sign = calculate_subnet_avg_sign(net, idx, false);
                let sign_str = format_sign(avg_sign);
                (format!("{name}{sign_str}"), format!(" (piece_sq_{idx})"))
            },
        )
    }
}

fn calculate_subnet_avg_sign(network: &QuantizedPolicyNetwork, idx: usize, is_square: bool) -> f64 {
    let mut total_weight: i64 = 0;
    let mut count = 0;

    let bias = if is_square {
        network.sq_subnet_bias(idx)
    } else {
        network.piece_sq_subnet_bias(idx)
    };

    for &bias_val in &bias.vals {
        if bias_val.abs() > SIGNIFICANT_WEIGHT_THRESHOLD {
            total_weight += bias_val as i64;
            count += 1;
        }
    }

    for feat_idx in 0..INPUT_SIZE {
        let weights = if is_square {
            network.sq_subnet_weight(idx, feat_idx)
        } else {
            network.piece_sq_subnet_weight(idx, feat_idx)
        };

        for &weight_val in &weights.vals {
            if weight_val.abs() > SIGNIFICANT_WEIGHT_THRESHOLD {
                total_weight += weight_val as i64;
                count += 1;
            }
        }
    }

    if count > 0 {
        total_weight as f64 / count as f64
    } else {
        0.0
    }
}

fn square_index_to_name(sq_idx: usize) -> String {
    if sq_idx >= 64 {
        return format!("invalid_{sq_idx}");
    }

    Square::from(sq_idx as u8).to_string()
}

fn format_sign(avg_sign: f64) -> &'static str {
    if avg_sign > 0.1 {
        " (+)"
    } else if avg_sign < -0.1 {
        " (-)"
    } else {
        " (~)"
    }
}

fn collect_all_weights<'a, GetBias, GetWeight>(
    count: usize,
    get_bias: GetBias,
    get_weight: GetWeight,
) -> Vec<i16>
where
    GetBias: Fn(usize) -> Accumulator<i16, ATTENTION_SIZE>,
    GetWeight: Fn(usize, usize) -> &'a Accumulator<i16, ATTENTION_SIZE>,
{
    let mut all_weights = Vec::new();
    for idx in 0..count {
        let bias = get_bias(idx);
        all_weights.extend_from_slice(&bias.vals);

        for feat_idx in 0..INPUT_SIZE {
            let weights = get_weight(idx, feat_idx);
            all_weights.extend_from_slice(&weights.vals);
        }
    }
    all_weights
}

fn decode_piece_sq_index(piece_sq_idx: usize) -> String {
    let square_idx = piece_sq_idx % 64;
    let bucket = piece_sq_idx / 64;

    let square_name = square_index_to_name(square_idx);

    let piece_info = match bucket {
        0 => "bad_see_pawn".to_string(),
        1 => "bad_see_knight".to_string(),
        2 => "bad_see_bishop".to_string(),
        3 => "bad_see_rook".to_string(),
        4 => "bad_see_queen".to_string(),
        5 => "king".to_string(),
        6 => "good_see_pawn".to_string(),
        7 => "good_see_knight".to_string(),
        8 => "good_see_bishop".to_string(),
        9 => "good_see_rook".to_string(),
        10 => "good_see_queen".to_string(),
        _ => format!("bucket_{bucket}"),
    };

    format!("{piece_info}_to_{square_name}")
}

fn analyze_attention_utilization(network: &QuantizedPolicyNetwork) {
    println!("\n--- Attention Node Utilization Analysis ---");

    let mut sq_attention_activity = [0u64; ATTENTION_SIZE];
    let mut piece_sq_attention_activity = [0u64; ATTENTION_SIZE];

    for idx in 0..network.sq_count() {
        let bias = network.sq_subnet_bias(idx);

        for (attention_idx, &bias_val) in bias.vals.iter().enumerate() {
            if bias_val.abs() > ATTENTION_ACTIVE_THRESHOLD {
                sq_attention_activity[attention_idx] += 1;
            }
        }

        for feat_idx in 0..INPUT_SIZE {
            let weights = network.sq_subnet_weight(idx, feat_idx);
            for (attention_idx, &weight_val) in weights.vals.iter().enumerate() {
                if weight_val.abs() > ATTENTION_ACTIVE_THRESHOLD {
                    sq_attention_activity[attention_idx] += 1;
                }
            }
        }
    }

    for idx in 0..network.piece_sq_count() {
        let bias = network.piece_sq_subnet_bias(idx);

        for (attention_idx, &bias_val) in bias.vals.iter().enumerate() {
            if bias_val.abs() > ATTENTION_ACTIVE_THRESHOLD {
                piece_sq_attention_activity[attention_idx] += 1;
            }
        }

        for feat_idx in 0..INPUT_SIZE {
            let weights = network.piece_sq_subnet_weight(idx, feat_idx);
            for (attention_idx, &weight_val) in weights.vals.iter().enumerate() {
                if weight_val.abs() > ATTENTION_ACTIVE_THRESHOLD {
                    piece_sq_attention_activity[attention_idx] += 1;
                }
            }
        }
    }

    let total_sq_connections = network.sq_count() * (INPUT_SIZE + 1);
    let total_piece_sq_connections = network.piece_sq_count() * (INPUT_SIZE + 1);

    println!("Square subnet attention node utilization:");
    for (attention_idx, &activity) in sq_attention_activity.iter().enumerate() {
        let utilization = 100.0 * activity as f64 / total_sq_connections as f64;
        println!(
            "  Node {attention_idx}: {utilization:.1}% active ({activity}/{total_sq_connections})"
        );
    }

    println!("\nPiece-square subnet attention node utilization:");
    for (attention_idx, &activity) in piece_sq_attention_activity.iter().enumerate() {
        let utilization = 100.0 * activity as f64 / total_piece_sq_connections as f64;
        println!("  Node {attention_idx}: {utilization:.1}% active ({activity}/{total_piece_sq_connections})");
    }

    println!("\nPiece-type attention breakdown:");
    let piece_names = [
        "bad_see_pawn",
        "bad_see_knight",
        "bad_see_bishop",
        "bad_see_rook",
        "bad_see_queen",
        "king",
        "good_see_pawn",
        "good_see_knight",
        "good_see_bishop",
        "good_see_rook",
        "good_see_queen",
    ];

    for (piece_idx, &piece_name) in piece_names.iter().enumerate() {
        let mut piece_activity = [0u64; ATTENTION_SIZE];
        let mut piece_connections = 0u64;

        for idx in (piece_idx * 64)..((piece_idx + 1) * 64).min(network.piece_sq_count()) {
            piece_connections += INPUT_SIZE as u64 + 1;

            let bias = network.piece_sq_subnet_bias(idx);
            for (attention_idx, &bias_val) in bias.vals.iter().enumerate() {
                if bias_val.abs() > ATTENTION_ACTIVE_THRESHOLD {
                    piece_activity[attention_idx] += 1;
                }
            }

            for feat_idx in 0..INPUT_SIZE {
                let weights = network.piece_sq_subnet_weight(idx, feat_idx);
                for (attention_idx, &weight_val) in weights.vals.iter().enumerate() {
                    if weight_val.abs() > ATTENTION_ACTIVE_THRESHOLD {
                        piece_activity[attention_idx] += 1;
                    }
                }
            }
        }

        let avg_utilization = piece_activity
            .iter()
            .map(|&x| x as f64 / piece_connections as f64)
            .sum::<f64>()
            / ATTENTION_SIZE as f64;
        let node_utils: Vec<String> = piece_activity
            .iter()
            .map(|&x| format!("{:.0}%", 100.0 * x as f64 / piece_connections as f64))
            .collect();
        println!(
            "  {}: {:.1}% average ({}) per node",
            piece_name,
            avg_utilization * 100.0,
            node_utils.join("/")
        );
    }

    let sq_underutilized: Vec<usize> = sq_attention_activity
        .iter()
        .enumerate()
        .filter(|(_, &activity)| {
            (activity as f64 / total_sq_connections as f64) < LOW_UTILIZATION_THRESHOLD
        })
        .map(|(idx, _)| idx)
        .collect();

    let piece_sq_underutilized: Vec<usize> = piece_sq_attention_activity
        .iter()
        .enumerate()
        .filter(|(_, &activity)| {
            (activity as f64 / total_piece_sq_connections as f64) < LOW_UTILIZATION_THRESHOLD
        })
        .map(|(idx, _)| idx)
        .collect();

    if !sq_underutilized.is_empty() {
        println!("\n⚠️  Square subnet underutilized attention nodes (<10%): {sq_underutilized:?}");
    }

    if !piece_sq_underutilized.is_empty() {
        println!("⚠️  Piece-square subnet underutilized attention nodes (<10%): {piece_sq_underutilized:?}");
    }

    let avg_sq_utilization = sq_attention_activity
        .iter()
        .map(|&x| x as f64 / total_sq_connections as f64)
        .sum::<f64>()
        / ATTENTION_SIZE as f64;
    let avg_piece_sq_utilization = piece_sq_attention_activity
        .iter()
        .map(|&x| x as f64 / total_piece_sq_connections as f64)
        .sum::<f64>()
        / ATTENTION_SIZE as f64;

    println!("\nAverage attention node utilization:");
    println!("  Square subnets: {:.1}%", avg_sq_utilization * 100.0);
    println!(
        "  Piece-square subnets: {:.1}%",
        avg_piece_sq_utilization * 100.0
    );
}

fn print_weight_stats(weights: &[i16]) {
    if weights.is_empty() {
        println!("  No weights to analyze");
        return;
    }

    let min_weight = *weights.iter().min().unwrap();
    let max_weight = *weights.iter().max().unwrap();
    let sum: i64 = weights.iter().map(|&w| w as i64).sum();
    let mean = sum as f64 / weights.len() as f64;

    let variance: f64 = weights
        .iter()
        .map(|&w| (w as f64 - mean).powi(2))
        .sum::<f64>()
        / weights.len() as f64;
    let std_dev = variance.sqrt();

    let zero_count = weights.iter().filter(|&&w| w == 0).count();
    let near_zero_count = weights.iter().filter(|&&w| w.abs() < 10).count();

    println!("  Min: {min_weight}, Max: {max_weight}, Mean: {mean:.2}, Std: {std_dev:.2}");
    let zero_pct = 100.0 * zero_count as f64 / weights.len() as f64;
    let near_zero_pct = 100.0 * near_zero_count as f64 / weights.len() as f64;
    println!("  Zero weights: {zero_count} ({zero_pct:.1}%)");
    println!("  Near-zero weights (abs < 10): {near_zero_count} ({near_zero_pct:.1}%)");
}
