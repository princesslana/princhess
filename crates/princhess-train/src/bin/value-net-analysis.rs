use princhess::chess::Square;
use princhess::quantized_value::{QuantizedValueNetwork, HIDDEN_SIZE, INPUT_SIZE};
use princhess::state::{NUMBER_KING_BUCKETS, NUMBER_POSITIONS};

type FeatureBucketMap = std::collections::HashMap<(usize, usize, bool), Vec<(usize, i64)>>;
use std::env;
use std::fs;

const DEAD_FEATURE_THRESHOLD: i32 = 100;
const SIGNIFICANT_WEIGHT_THRESHOLD: i16 = 5;
const LOW_UTILIZATION_THRESHOLD: f64 = 0.1;

// Helper functions to eliminate redundant mappings
fn king_bucket_name(bucket: usize, short: bool) -> &'static str {
    match bucket {
        0 => {
            if short {
                "Corner"
            } else {
                "King corner"
            }
        }
        1 => {
            if short {
                "Center"
            } else {
                "King center"
            }
        }
        2 => {
            if short {
                "Other"
            } else {
                "King other"
            }
        }
        _ => "Unknown",
    }
}

fn threat_bucket_name(bucket: usize) -> &'static str {
    match bucket {
        0 => "Safe",
        1 => "Defended",
        2 => "Threatened",
        3 => "Both",
        _ => "Unknown",
    }
}

fn piece_name(piece_idx: usize) -> &'static str {
    match piece_idx {
        0 => "pawn",
        1 => "knight",
        2 => "bishop",
        3 => "rook",
        4 => "queen",
        5 => "king",
        _ => "unknown",
    }
}

fn main() {
    let mut args = env::args();
    args.next();

    let network_path = args
        .next()
        .expect("Usage: value-analysis <path-to-value.bin>");

    println!("Analyzing value network: {network_path}");
    println!();

    let network_bytes = fs::read(&network_path)
        .unwrap_or_else(|e| panic!("Failed to read network file {network_path}: {e}"));

    let network: &QuantizedValueNetwork =
        unsafe { &*(network_bytes.as_ptr() as *const QuantizedValueNetwork) };

    analyze_network_weights(network);
}

fn analyze_network_weights(network: &QuantizedValueNetwork) {
    println!("=== VALUE NETWORK WEIGHT ANALYSIS ===");

    println!("\n--- Side-to-Move (STM) Layer Analysis ---");
    let stm_weights = collect_feature_weights(network, true);
    let stm_bias = collect_bias_weights(network, true);
    analyze_feature_layer(&stm_weights, &stm_bias, "STM");

    println!("\n--- Non-Side-to-Move (NSTM) Layer Analysis ---");
    let nstm_weights = collect_feature_weights(network, false);
    let nstm_bias = collect_bias_weights(network, false);
    analyze_feature_layer(&nstm_weights, &nstm_bias, "NSTM");

    println!("\n--- Output Layer Analysis ---");
    analyze_output_layer(network);

    println!("\n--- Hidden Node Utilization Analysis ---");
    analyze_hidden_node_utilization(network);

    println!("\n--- Hidden Layer Size Analysis ---");
    analyze_hidden_layer_size(network);

    println!("\n--- Feature Importance Analysis ---");
    analyze_feature_importance(network);

    println!("\n--- Bucket Differentiation Analysis ---");
    analyze_bucket_differentiation(network);

    println!("\n--- Bucket Similarity Matrix ---");
    analyze_bucket_similarity_matrix(network);

    println!("\n--- Network Architecture Summary ---");
    println!("Input features: {INPUT_SIZE}");
    println!("Hidden size: {HIDDEN_SIZE}");
    println!("Total parameters: {}", calculate_total_parameters());
}

fn collect_feature_weights(network: &QuantizedValueNetwork, is_stm: bool) -> Vec<i16> {
    let mut weights = Vec::new();

    for feat_idx in 0..INPUT_SIZE {
        let feature_weights = if is_stm {
            &network.stm_weight(feat_idx).vals
        } else {
            &network.nstm_weight(feat_idx).vals
        };
        weights.extend_from_slice(feature_weights);
    }

    weights
}

fn collect_bias_weights(network: &QuantizedValueNetwork, is_stm: bool) -> Vec<i16> {
    let bias = if is_stm {
        &network.stm_bias().vals
    } else {
        &network.nstm_bias().vals
    };
    bias.to_vec()
}

fn analyze_feature_layer(weights: &[i16], bias: &[i16], layer_name: &str) {
    println!("\n{layer_name} layer statistics:");

    println!("Feature weights:");
    print_weight_stats(weights);

    println!("Bias weights:");
    print_weight_stats(bias);

    let dead_features = count_dead_features(weights, bias);
    if dead_features > 0 {
        println!(
            "⚠️  {dead_features} potentially dead hidden nodes detected in {layer_name} layer"
        );
    }
}

fn analyze_output_layer(network: &QuantizedValueNetwork) {
    let mut stm_output_weights = Vec::new();
    let mut nstm_output_weights = Vec::new();

    stm_output_weights.extend_from_slice(&network.output_weight(0).vals);
    nstm_output_weights.extend_from_slice(&network.output_weight(1).vals);

    println!("STM output weights:");
    print_weight_stats(&stm_output_weights);

    println!("NSTM output weights:");
    print_weight_stats(&nstm_output_weights);

    println!("Output bias: {}", network.output_bias());

    let stm_zero_count = stm_output_weights
        .iter()
        .filter(|&&w| w.abs() < SIGNIFICANT_WEIGHT_THRESHOLD)
        .count();
    let nstm_zero_count = nstm_output_weights
        .iter()
        .filter(|&&w| w.abs() < SIGNIFICANT_WEIGHT_THRESHOLD)
        .count();

    if stm_zero_count > HIDDEN_SIZE / 10 {
        println!("⚠️  Many insignificant STM output weights: {stm_zero_count}/{HIDDEN_SIZE}");
    }

    if nstm_zero_count > HIDDEN_SIZE / 10 {
        println!("⚠️  Many insignificant NSTM output weights: {nstm_zero_count}/{HIDDEN_SIZE}");
    }
}

fn analyze_hidden_node_utilization(network: &QuantizedValueNetwork) {
    let mut node_activity = vec![0u32; HIDDEN_SIZE];

    for feat_idx in 0..INPUT_SIZE {
        for (hidden_idx, &weight) in network.stm_weight(feat_idx).vals.iter().enumerate() {
            if weight.abs() > SIGNIFICANT_WEIGHT_THRESHOLD {
                node_activity[hidden_idx] += 1;
            }
        }

        for (hidden_idx, &weight) in network.nstm_weight(feat_idx).vals.iter().enumerate() {
            if weight.abs() > SIGNIFICANT_WEIGHT_THRESHOLD {
                node_activity[hidden_idx] += 1;
            }
        }
    }

    for (hidden_idx, &bias) in network.stm_bias().vals.iter().enumerate() {
        if bias.abs() > SIGNIFICANT_WEIGHT_THRESHOLD {
            node_activity[hidden_idx] += 1;
        }
    }

    for (hidden_idx, &bias) in network.nstm_bias().vals.iter().enumerate() {
        if bias.abs() > SIGNIFICANT_WEIGHT_THRESHOLD {
            node_activity[hidden_idx] += 1;
        }
    }

    for (hidden_idx, &weight) in network.output_weight(0).vals.iter().enumerate() {
        if weight.abs() > SIGNIFICANT_WEIGHT_THRESHOLD {
            node_activity[hidden_idx] += 1;
        }
    }

    for (hidden_idx, &weight) in network.output_weight(1).vals.iter().enumerate() {
        if weight.abs() > SIGNIFICANT_WEIGHT_THRESHOLD {
            node_activity[hidden_idx] += 1;
        }
    }

    let total_connections = (INPUT_SIZE * 2 + 2 + 2) as u32; // 2 feature layers, 2 biases, 2 output connections
    let underutilized_nodes: Vec<usize> = node_activity
        .iter()
        .enumerate()
        .filter(|(_, &activity)| {
            (activity as f64 / total_connections as f64) < LOW_UTILIZATION_THRESHOLD
        })
        .map(|(idx, _)| idx)
        .collect();

    let avg_utilization = node_activity
        .iter()
        .map(|&x| x as f64 / total_connections as f64)
        .sum::<f64>()
        / HIDDEN_SIZE as f64;

    println!(
        "Average hidden node utilization: {:.1}%",
        avg_utilization * 100.0
    );

    if !underutilized_nodes.is_empty() {
        println!(
            "⚠️  {} underutilized hidden nodes (<10%): {:?}",
            underutilized_nodes.len(),
            &underutilized_nodes[..underutilized_nodes.len().min(10)]
        );
        if underutilized_nodes.len() > 10 {
            println!("    ... and {} more", underutilized_nodes.len() - 10);
        }
    }

    let highly_utilized = node_activity
        .iter()
        .enumerate()
        .filter(|(_, &activity)| (activity as f64 / total_connections as f64) > 0.8)
        .count();

    println!("Highly utilized nodes (>80%): {highly_utilized}/{HIDDEN_SIZE}");
}

fn count_dead_features(weights: &[i16], bias: &[i16]) -> usize {
    let mut dead_count = 0;

    for (hidden_idx, &bias_val) in bias.iter().enumerate().take(HIDDEN_SIZE) {
        let mut total_magnitude = bias_val.abs() as i32;

        for feat_idx in 0..INPUT_SIZE {
            let weight_idx = feat_idx * HIDDEN_SIZE + hidden_idx;
            if weight_idx < weights.len() {
                total_magnitude += weights[weight_idx].abs() as i32;
            }
        }

        if total_magnitude < DEAD_FEATURE_THRESHOLD {
            dead_count += 1;
        }
    }

    dead_count
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

fn analyze_feature_importance(network: &QuantizedValueNetwork) {
    let mut feature_importance = vec![0i64; INPUT_SIZE];

    // Calculate total weight magnitude for each feature across both STM and NSTM
    for (feat_idx, feature_mag) in feature_importance.iter_mut().enumerate().take(INPUT_SIZE) {
        let mut total_magnitude = 0i64;

        // STM weights
        for &weight in &network.stm_weight(feat_idx).vals {
            total_magnitude += weight.abs() as i64;
        }

        // NSTM weights
        for &weight in &network.nstm_weight(feat_idx).vals {
            total_magnitude += weight.abs() as i64;
        }

        *feature_mag = total_magnitude;
    }

    // Sort by importance
    let mut indexed_features: Vec<(usize, i64)> =
        feature_importance.into_iter().enumerate().collect();
    indexed_features.sort_by_key(|(_, importance)| std::cmp::Reverse(*importance));

    println!("Top 10 most important features:");
    for (i, (feat_idx, importance)) in indexed_features.iter().take(10).enumerate() {
        let desc = decode_feature_index(*feat_idx);
        println!("  {}: {} (magnitude: {})", i + 1, desc, importance);
    }

    println!("\nBottom 10 least important features:");
    for (i, (feat_idx, importance)) in indexed_features.iter().rev().take(10).enumerate() {
        let desc = decode_feature_index(*feat_idx);
        println!("  {}: {} (magnitude: {})", i + 1, desc, importance);
    }

    let zero_features = indexed_features
        .iter()
        .filter(|&(_, imp)| *imp < 100)
        .count();
    if zero_features > 0 {
        println!("\n⚠️  {zero_features} features with very low importance (<100 total magnitude)");
    }
}

fn analyze_bucket_differentiation(network: &QuantizedValueNetwork) {
    // For feature-by-feature analysis, we need to group features by their logical identity
    // Each "logical feature" appears in multiple buckets (king + threat combinations)

    // Map from (piece, square, side) -> Vec<(bucket_combo, importance)>
    let mut feature_buckets: std::collections::HashMap<(usize, usize, bool), Vec<(usize, i64)>> =
        std::collections::HashMap::new();

    for feat_idx in 0..INPUT_SIZE {
        let bucket = feat_idx / NUMBER_POSITIONS;
        let position = feat_idx % NUMBER_POSITIONS;

        let is_opponent = position >= 384;
        let piece_pos = position % 384;
        let piece_idx = piece_pos / 64;
        let square_idx = piece_pos % 64;

        let logical_feature = (piece_idx, square_idx, is_opponent);

        let mut total_magnitude = 0i64;

        // Collect weights from both STM and NSTM
        for &weight in &network.stm_weight(feat_idx).vals {
            total_magnitude += weight.abs() as i64;
        }
        for &weight in &network.nstm_weight(feat_idx).vals {
            total_magnitude += weight.abs() as i64;
        }

        feature_buckets
            .entry(logical_feature)
            .or_default()
            .push((bucket, total_magnitude));
    }

    // Calculate differentiation stats for each logical feature
    let mut differentiation_ratios = Vec::new();
    let mut zero_bucket_stats = Vec::new();

    for ((piece_idx, square_idx, is_opponent), bucket_values) in &feature_buckets {
        if bucket_values.len() > 1 {
            let importances: Vec<i64> = bucket_values
                .iter()
                .map(|(_, importance)| *importance)
                .collect();
            let max_importance = *importances.iter().max().unwrap();
            let min_importance = *importances.iter().min().unwrap();

            let zero_bucket_count = importances.iter().filter(|&&imp| imp == 0).count();
            let total_buckets = importances.len();

            zero_bucket_stats.push((zero_bucket_count, total_buckets));

            let ratio = if min_importance > 0 {
                max_importance as f64 / min_importance as f64
            } else if max_importance > 0 {
                f64::INFINITY
            } else {
                1.0
            };

            differentiation_ratios.push((
                (*piece_idx, *square_idx, *is_opponent),
                ratio,
                bucket_values.clone(),
            ));
        }
    }

    // Sort by differentiation ratio
    differentiation_ratios
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Feature-by-feature bucket differentiation analysis:");
    println!("(Comparing same logical feature across different king/threat bucket combinations)\n");

    // Overall statistics
    let finite_ratios: Vec<f64> = differentiation_ratios
        .iter()
        .map(|(_, ratio, _)| *ratio)
        .filter(|r| r.is_finite())
        .collect();

    if !finite_ratios.is_empty() {
        let avg_ratio = finite_ratios.iter().sum::<f64>() / finite_ratios.len() as f64;
        let median_ratio = {
            let mut sorted = finite_ratios.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };

        println!("Differentiation statistics:");
        println!("  Average ratio: {avg_ratio:.2}x");
        println!("  Median ratio: {median_ratio:.2}x");
        println!("  Features analyzed: {}", finite_ratios.len());

        let high_diff_count = finite_ratios.iter().filter(|&&r| r > 3.0).count();
        let low_diff_count = finite_ratios.iter().filter(|&&r| r < 1.5).count();

        println!(
            "  High differentiation (>3x): {} features ({:.1}%)",
            high_diff_count,
            100.0 * high_diff_count as f64 / finite_ratios.len() as f64
        );
        println!(
            "  Low differentiation (<1.5x): {} features ({:.1}%)",
            low_diff_count,
            100.0 * low_diff_count as f64 / finite_ratios.len() as f64
        );

        // Zero bucket statistics
        let total_zero_buckets: usize = zero_bucket_stats.iter().map(|(zeros, _)| zeros).sum();
        let total_all_buckets: usize = zero_bucket_stats.iter().map(|(_, total)| total).sum();
        let features_with_zero_buckets = zero_bucket_stats
            .iter()
            .filter(|(zeros, _)| *zeros > 0)
            .count();
        let features_with_all_zero_buckets = zero_bucket_stats
            .iter()
            .filter(|(zeros, total)| zeros == total)
            .count();

        println!("\nZero importance statistics:");
        println!("  Total feature-bucket combinations: {total_all_buckets}");
        println!(
            "  Zero importance combinations: {} ({:.1}%)",
            total_zero_buckets,
            100.0 * total_zero_buckets as f64 / total_all_buckets as f64
        );
        println!(
            "  Features with some zero buckets: {} ({:.1}%)",
            features_with_zero_buckets,
            100.0 * features_with_zero_buckets as f64 / finite_ratios.len() as f64
        );
        println!(
            "  Features with all zero buckets: {} ({:.1}%)",
            features_with_all_zero_buckets,
            100.0 * features_with_all_zero_buckets as f64 / finite_ratios.len() as f64
        );

        // Per-bucket analysis
        analyze_per_bucket_stats(&feature_buckets);
    }

    println!("\nTop 10 most differentiated features:");
    for (i, ((piece_idx, square_idx, is_opponent), ratio, bucket_values)) in
        differentiation_ratios.iter().take(10).enumerate()
    {
        let piece_name = piece_name(*piece_idx);
        let square = Square::from(*square_idx as u8).to_string();
        let side = if *is_opponent { "opp" } else { "our" };

        println!(
            "  {}: {}_{}_{}  {:.2}x",
            i + 1,
            side,
            piece_name,
            square,
            ratio
        );

        // Show the bucket breakdown
        let mut sorted_buckets = bucket_values.clone();
        sorted_buckets.sort_by_key(|(_, importance)| std::cmp::Reverse(*importance));
        let bucket_info: Vec<String> = sorted_buckets
            .iter()
            .map(|(bucket, importance)| {
                let king_bucket = bucket % NUMBER_KING_BUCKETS;
                let threat_bucket = bucket / NUMBER_KING_BUCKETS;
                format!("K{king_bucket}T{threat_bucket}:{importance}")
            })
            .collect();
        println!("     {}", bucket_info.join(", "));
    }

    println!("\nBottom 10 least differentiated features:");
    for (i, ((piece_idx, square_idx, is_opponent), ratio, _bucket_values)) in
        differentiation_ratios.iter().rev().take(10).enumerate()
    {
        if !ratio.is_finite() {
            continue;
        }

        let piece_name = piece_name(*piece_idx);
        let square = Square::from(*square_idx as u8).to_string();
        let side = if *is_opponent { "opp" } else { "our" };

        println!(
            "  {}: {}_{}_{}  {:.2}x",
            i + 1,
            side,
            piece_name,
            square,
            ratio
        );
    }
}

fn decode_feature_index(feat_idx: usize) -> String {
    let bucket = feat_idx / NUMBER_POSITIONS;
    let position = feat_idx % NUMBER_POSITIONS;

    let king_bucket = bucket % NUMBER_KING_BUCKETS;
    let threat_bucket = bucket / NUMBER_KING_BUCKETS;

    let is_opponent = position >= 384;
    let piece_pos = position % 384;
    let piece_idx = piece_pos / 64;
    let square_idx = piece_pos % 64;

    let piece_name = piece_name(piece_idx);

    let square = Square::from(square_idx as u8).to_string();

    let side = if is_opponent { "opp" } else { "our" };

    let king_desc = match king_bucket {
        0 => "king_corner",
        1 => "king_center",
        2 => "king_other",
        _ => "king_unknown",
    };

    let threat_desc = match threat_bucket {
        0 => "safe",
        1 => "defended",
        2 => "threatened",
        3 => "both",
        _ => "threat_unknown",
    };

    format!("{side}_{piece_name}_{square}_{king_desc}__{threat_desc}")
}

fn analyze_hidden_layer_size(network: &QuantizedValueNetwork) {
    let mut node_total_magnitude = vec![0i64; HIDDEN_SIZE];

    // Calculate total magnitude for each hidden node across all connections
    for feat_idx in 0..INPUT_SIZE {
        for (hidden_idx, &weight) in network.stm_weight(feat_idx).vals.iter().enumerate() {
            node_total_magnitude[hidden_idx] += weight.abs() as i64;
        }

        for (hidden_idx, &weight) in network.nstm_weight(feat_idx).vals.iter().enumerate() {
            node_total_magnitude[hidden_idx] += weight.abs() as i64;
        }
    }

    // Add bias contributions
    for (hidden_idx, &bias) in network.stm_bias().vals.iter().enumerate() {
        node_total_magnitude[hidden_idx] += bias.abs() as i64;
    }

    for (hidden_idx, &bias) in network.nstm_bias().vals.iter().enumerate() {
        node_total_magnitude[hidden_idx] += bias.abs() as i64;
    }

    // Add output weight contributions
    for (hidden_idx, &weight) in network.output_weight(0).vals.iter().enumerate() {
        node_total_magnitude[hidden_idx] += weight.abs() as i64;
    }

    for (hidden_idx, &weight) in network.output_weight(1).vals.iter().enumerate() {
        node_total_magnitude[hidden_idx] += weight.abs() as i64;
    }

    // Sort nodes by total magnitude
    let mut indexed_nodes: Vec<(usize, i64)> =
        node_total_magnitude.into_iter().enumerate().collect();
    indexed_nodes.sort_by_key(|(_, magnitude)| std::cmp::Reverse(*magnitude));

    // Calculate statistics
    let magnitudes: Vec<i64> = indexed_nodes.iter().map(|(_, mag)| *mag).collect();
    let total_magnitude: i64 = magnitudes.iter().sum();
    let avg_magnitude = total_magnitude as f64 / HIDDEN_SIZE as f64;
    let max_magnitude = *magnitudes.iter().max().unwrap();
    let min_magnitude = *magnitudes.iter().min().unwrap();

    // Count nodes by utilization level
    let dead_nodes = magnitudes.iter().filter(|&&mag| mag == 0).count();
    let very_low_nodes = magnitudes
        .iter()
        .filter(|&&mag| mag > 0 && mag < (avg_magnitude * 0.1) as i64)
        .count();
    let low_nodes = magnitudes
        .iter()
        .filter(|&&mag| mag >= (avg_magnitude * 0.1) as i64 && mag < (avg_magnitude * 0.5) as i64)
        .count();
    let medium_nodes = magnitudes
        .iter()
        .filter(|&&mag| mag >= (avg_magnitude * 0.5) as i64 && mag < (avg_magnitude * 1.5) as i64)
        .count();
    let high_nodes = magnitudes
        .iter()
        .filter(|&&mag| mag >= (avg_magnitude * 1.5) as i64)
        .count();

    println!("Hidden layer utilization statistics:");
    println!("  Total hidden nodes: {HIDDEN_SIZE}");
    println!("  Average node magnitude: {avg_magnitude:.0}");
    println!("  Range: {min_magnitude} - {max_magnitude}");
    println!(
        "  Max/Min ratio: {:.2}x",
        max_magnitude as f64 / min_magnitude.max(1) as f64
    );

    println!("\nNode utilization distribution:");
    println!(
        "  Dead nodes (0 magnitude): {} ({:.1}%)",
        dead_nodes,
        100.0 * dead_nodes as f64 / HIDDEN_SIZE as f64
    );
    println!(
        "  Very low (<10% avg): {} ({:.1}%)",
        very_low_nodes,
        100.0 * very_low_nodes as f64 / HIDDEN_SIZE as f64
    );
    println!(
        "  Low (10-50% avg): {} ({:.1}%)",
        low_nodes,
        100.0 * low_nodes as f64 / HIDDEN_SIZE as f64
    );
    println!(
        "  Medium (50-150% avg): {} ({:.1}%)",
        medium_nodes,
        100.0 * medium_nodes as f64 / HIDDEN_SIZE as f64
    );
    println!(
        "  High (>150% avg): {} ({:.1}%)",
        high_nodes,
        100.0 * high_nodes as f64 / HIDDEN_SIZE as f64
    );

    // Show top and bottom performing nodes
    println!("\nTop 10 most utilized hidden nodes:");
    for (i, (node_idx, magnitude)) in indexed_nodes.iter().take(10).enumerate() {
        let percentage = 100.0 * *magnitude as f64 / total_magnitude as f64;
        println!(
            "  {}: Node {} - magnitude: {} ({:.2}% of total)",
            i + 1,
            node_idx,
            magnitude,
            percentage
        );
    }

    println!("\nBottom 10 least utilized hidden nodes:");
    for (i, (node_idx, magnitude)) in indexed_nodes.iter().rev().take(10).enumerate() {
        let percentage = 100.0 * *magnitude as f64 / total_magnitude as f64;
        println!(
            "  {}: Node {} - magnitude: {} ({:.2}% of total)",
            i + 1,
            node_idx,
            magnitude,
            percentage
        );
    }

    // Cumulative analysis - what percentage of nodes contain X% of total magnitude
    let mut cumulative_magnitude = 0i64;
    let mut nodes_for_50pct = 0;
    let mut nodes_for_80pct = 0;
    let mut nodes_for_95pct = 0;

    for (_, magnitude) in &indexed_nodes {
        cumulative_magnitude += magnitude;
        let cumulative_pct = 100.0 * cumulative_magnitude as f64 / total_magnitude as f64;

        if nodes_for_50pct == 0 && cumulative_pct >= 50.0 {
            nodes_for_50pct = cumulative_magnitude as usize / *magnitude as usize;
        }
        if nodes_for_80pct == 0 && cumulative_pct >= 80.0 {
            nodes_for_80pct = cumulative_magnitude as usize / *magnitude as usize;
        }
        if nodes_for_95pct == 0 && cumulative_pct >= 95.0 {
            nodes_for_95pct = cumulative_magnitude as usize / *magnitude as usize;
        }
    }

    // Fix the cumulative calculation
    cumulative_magnitude = 0;
    nodes_for_50pct = 0;
    nodes_for_80pct = 0;
    nodes_for_95pct = 0;

    for (count, (_, magnitude)) in indexed_nodes.iter().enumerate() {
        cumulative_magnitude += magnitude;
        let cumulative_pct = 100.0 * cumulative_magnitude as f64 / total_magnitude as f64;

        if nodes_for_50pct == 0 && cumulative_pct >= 50.0 {
            nodes_for_50pct = count + 1;
        }
        if nodes_for_80pct == 0 && cumulative_pct >= 80.0 {
            nodes_for_80pct = count + 1;
        }
        if nodes_for_95pct == 0 && cumulative_pct >= 95.0 {
            nodes_for_95pct = count + 1;
        }
    }

    println!("\nCumulative node importance:");
    println!(
        "  {} nodes ({:.1}%) contain 50% of total magnitude",
        nodes_for_50pct,
        100.0 * nodes_for_50pct as f64 / HIDDEN_SIZE as f64
    );
    println!(
        "  {} nodes ({:.1}%) contain 80% of total magnitude",
        nodes_for_80pct,
        100.0 * nodes_for_80pct as f64 / HIDDEN_SIZE as f64
    );
    println!(
        "  {} nodes ({:.1}%) contain 95% of total magnitude",
        nodes_for_95pct,
        100.0 * nodes_for_95pct as f64 / HIDDEN_SIZE as f64
    );

    // Architecture recommendations
    let underutilized = dead_nodes + very_low_nodes;
    if underutilized > HIDDEN_SIZE / 10 {
        println!(
            "\n⚠️  {:.1}% of hidden nodes are underutilized - consider reducing hidden layer size",
            100.0 * underutilized as f64 / HIDDEN_SIZE as f64
        );
    }

    if nodes_for_80pct < HIDDEN_SIZE / 2 {
        println!("⚠️  80% of network capacity is concentrated in {:.1}% of nodes - potential for significant pruning", 
            100.0 * nodes_for_80pct as f64 / HIDDEN_SIZE as f64);
    }
}

fn analyze_per_bucket_stats(feature_buckets: &FeatureBucketMap) {
    println!("\nPer-bucket analysis:");

    // Collect stats for each bucket combination
    let mut bucket_stats: std::collections::HashMap<usize, Vec<i64>> =
        std::collections::HashMap::new();

    for bucket_values in feature_buckets.values() {
        for &(bucket, importance) in bucket_values {
            bucket_stats.entry(bucket).or_default().push(importance);
        }
    }

    // Sort buckets by ID for consistent ordering
    let mut sorted_buckets: Vec<_> = bucket_stats.into_iter().collect();
    sorted_buckets.sort_by_key(|(bucket_id, _)| *bucket_id);

    for (bucket_id, importances) in sorted_buckets {
        let king_bucket = bucket_id % NUMBER_KING_BUCKETS;
        let threat_bucket = bucket_id / NUMBER_KING_BUCKETS;

        let king_desc = king_bucket_name(king_bucket, false);
        let threat_desc = threat_bucket_name(threat_bucket);

        let total_features = importances.len();
        let zero_features = importances.iter().filter(|&&imp| imp == 0).count();
        let zero_percentage = 100.0 * zero_features as f64 / total_features as f64;

        let non_zero_importances: Vec<_> = importances
            .iter()
            .filter(|&&imp| imp > 0)
            .cloned()
            .collect();

        if !non_zero_importances.is_empty() {
            let avg_importance =
                non_zero_importances.iter().sum::<i64>() as f64 / non_zero_importances.len() as f64;
            let max_importance = *non_zero_importances.iter().max().unwrap();
            let min_importance = *non_zero_importances.iter().min().unwrap();

            println!(
                "  K{king_bucket}T{threat_bucket} ({king_desc} + {threat_desc}): {total_features} features, {zero_percentage:.1}% zero, avg: {avg_importance:.0}, range: {min_importance}-{max_importance}"
            );
        } else {
            println!(
                "  K{king_bucket}T{threat_bucket} ({king_desc} + {threat_desc}): {total_features} features, {zero_percentage:.1}% zero, NO NON-ZERO FEATURES"
            );
        }
    }

    // Summary statistics comparing buckets
    println!("\nBucket comparison:");

    // Group by king bucket
    let mut king_bucket_zeros = [0usize; NUMBER_KING_BUCKETS];
    let mut king_bucket_totals = [0usize; NUMBER_KING_BUCKETS];
    let mut king_bucket_avg_importance = [0.0f64; NUMBER_KING_BUCKETS];
    let mut king_bucket_counts = [0usize; NUMBER_KING_BUCKETS];

    for bucket_values in feature_buckets.values() {
        for &(bucket, importance) in bucket_values {
            let king_bucket = bucket % NUMBER_KING_BUCKETS;
            king_bucket_totals[king_bucket] += 1;
            if importance == 0 {
                king_bucket_zeros[king_bucket] += 1;
            } else {
                king_bucket_avg_importance[king_bucket] += importance as f64;
                king_bucket_counts[king_bucket] += 1;
            }
        }
    }

    for (i, (&zeros, &totals)) in king_bucket_zeros
        .iter()
        .zip(king_bucket_totals.iter())
        .enumerate()
    {
        let zero_pct = 100.0 * zeros as f64 / totals as f64;
        let avg_imp = if king_bucket_counts[i] > 0 {
            king_bucket_avg_importance[i] / king_bucket_counts[i] as f64
        } else {
            0.0
        };
        let king_desc = king_bucket_name(i, false);
        println!("  {king_desc}: {zero_pct:.1}% zero features, avg importance: {avg_imp:.0}");
    }

    // Group by threat bucket
    let mut threat_bucket_zeros = [0usize; 4]; // NUMBER_THREAT_BUCKETS
    let mut threat_bucket_totals = [0usize; 4];
    let mut threat_bucket_avg_importance = [0.0f64; 4];
    let mut threat_bucket_counts = [0usize; 4];

    for bucket_values in feature_buckets.values() {
        for &(bucket, importance) in bucket_values {
            let threat_bucket = bucket / NUMBER_KING_BUCKETS;
            if threat_bucket < 4 {
                threat_bucket_totals[threat_bucket] += 1;
                if importance == 0 {
                    threat_bucket_zeros[threat_bucket] += 1;
                } else {
                    threat_bucket_avg_importance[threat_bucket] += importance as f64;
                    threat_bucket_counts[threat_bucket] += 1;
                }
            }
        }
    }

    for (i, (&zeros, &totals)) in threat_bucket_zeros
        .iter()
        .zip(threat_bucket_totals.iter())
        .enumerate()
    {
        let zero_pct = 100.0 * zeros as f64 / totals as f64;
        let avg_imp = if threat_bucket_counts[i] > 0 {
            threat_bucket_avg_importance[i] / threat_bucket_counts[i] as f64
        } else {
            0.0
        };
        let threat_desc = threat_bucket_name(i);
        println!("  {threat_desc}: {zero_pct:.1}% zero features, avg importance: {avg_imp:.0}");
    }
}

fn analyze_bucket_similarity_matrix(network: &QuantizedValueNetwork) {
    const NUM_BUCKETS: usize = NUMBER_KING_BUCKETS * 4; // 3 king buckets * 4 threat buckets = 12

    // Collect feature importance for each bucket
    let mut bucket_features = vec![vec![0i64; NUMBER_POSITIONS]; NUM_BUCKETS];

    for feat_idx in 0..INPUT_SIZE {
        let bucket = feat_idx / NUMBER_POSITIONS;
        let position = feat_idx % NUMBER_POSITIONS;

        if bucket < NUM_BUCKETS {
            let mut total_magnitude = 0i64;

            // Collect weights from both STM and NSTM
            for &weight in &network.stm_weight(feat_idx).vals {
                total_magnitude += weight.abs() as i64;
            }
            for &weight in &network.nstm_weight(feat_idx).vals {
                total_magnitude += weight.abs() as i64;
            }

            bucket_features[bucket][position] = total_magnitude;
        }
    }

    // Calculate similarity matrix using cosine similarity
    let mut similarity_matrix = vec![vec![0.0f64; NUM_BUCKETS]; NUM_BUCKETS];

    for i in 0..NUM_BUCKETS {
        for j in 0..NUM_BUCKETS {
            if i == j {
                similarity_matrix[i][j] = 1.0;
            } else {
                similarity_matrix[i][j] =
                    cosine_similarity(&bucket_features[i], &bucket_features[j]);
            }
        }
    }

    // Print bucket labels
    println!("Bucket similarity matrix (cosine similarity):");
    println!("Buckets: K0T0, K1T0, K2T0, K0T1, K1T1, K2T1, K0T2, K1T2, K2T2, K0T3, K1T3, K2T3");
    println!(
        "Legend: K0=Corner, K1=Center, K2=Other; T0=Safe, T1=Defended, T2=Threatened, T3=Both\n"
    );

    // Print column headers
    print!("      ");
    for j in 0..NUM_BUCKETS {
        let k = j % NUMBER_KING_BUCKETS;
        let t = j / NUMBER_KING_BUCKETS;
        print!(" K{k}T{t}");
    }
    println!();

    // Print matrix with row labels
    for (i, row) in similarity_matrix.iter().enumerate().take(NUM_BUCKETS) {
        let k = i % NUMBER_KING_BUCKETS;
        let t = i / NUMBER_KING_BUCKETS;
        print!("K{k}T{t} ");

        for value in row.iter().take(NUM_BUCKETS) {
            print!(" {value:.2}");
        }
        println!();
    }

    // Find most and least similar bucket pairs
    let mut similarities = Vec::new();
    for (i, row) in similarity_matrix.iter().enumerate().take(NUM_BUCKETS) {
        for (j_idx, value) in row.iter().enumerate().skip(i + 1) {
            similarities.push((i, j_idx, *value));
        }
    }

    similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nMost similar bucket pairs:");
    for (i, j, sim) in similarities.iter().take(10) {
        let k1 = i % NUMBER_KING_BUCKETS;
        let t1 = i / NUMBER_KING_BUCKETS;
        let k2 = j % NUMBER_KING_BUCKETS;
        let t2 = j / NUMBER_KING_BUCKETS;
        println!("  K{k1}T{t1} ↔ K{k2}T{t2}: {sim:.3}");
    }

    println!("\nLeast similar bucket pairs:");
    for (i, j, sim) in similarities.iter().rev().take(10) {
        let k1 = i % NUMBER_KING_BUCKETS;
        let t1 = i / NUMBER_KING_BUCKETS;
        let k2 = j % NUMBER_KING_BUCKETS;
        let t2 = j / NUMBER_KING_BUCKETS;
        println!("  K{k1}T{t1} ↔ K{k2}T{t2}: {sim:.3}");
    }

    // Calculate average similarity within king bucket groups and threat bucket groups
    let mut king_bucket_sims = vec![Vec::new(); NUMBER_KING_BUCKETS];
    let mut threat_bucket_sims = vec![Vec::new(); 4];

    for (i, row) in similarity_matrix.iter().enumerate().take(NUM_BUCKETS) {
        for (j_idx, value) in row.iter().enumerate().skip(i + 1) {
            let k1 = i % NUMBER_KING_BUCKETS;
            let t1 = i / NUMBER_KING_BUCKETS;
            let k2 = j_idx % NUMBER_KING_BUCKETS;
            let t2 = j_idx / NUMBER_KING_BUCKETS;

            // Same king bucket, different threat bucket
            if k1 == k2 && t1 != t2 {
                king_bucket_sims[k1].push(*value);
            }

            // Same threat bucket, different king bucket
            if t1 == t2 && k1 != k2 {
                threat_bucket_sims[t1].push(*value);
            }
        }
    }

    println!("\nIntra-group similarity averages:");
    for (k, sims) in king_bucket_sims.iter().enumerate() {
        if !sims.is_empty() {
            let avg = sims.iter().sum::<f64>() / sims.len() as f64;
            let king_desc = king_bucket_name(k, true);
            println!(
                "  King bucket {k} ({king_desc}): {avg:.3} avg similarity across threat buckets"
            );
        }
    }

    for (t, sims) in threat_bucket_sims.iter().enumerate() {
        let threat_desc = threat_bucket_name(t);
        if !sims.is_empty() {
            let avg = sims.iter().sum::<f64>() / sims.len() as f64;
            println!(
                "  Threat bucket {t} ({threat_desc}): {avg:.3} avg similarity across king buckets"
            );
        } else {
            println!(
                "  Threat bucket {t} ({threat_desc}): no similarity data (insufficient bucket pairs)"
            );
        }
    }
}

fn cosine_similarity(vec1: &[i64], vec2: &[i64]) -> f64 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }

    let dot_product: f64 = vec1
        .iter()
        .zip(vec2.iter())
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();

    let norm1: f64 = vec1.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm2: f64 = vec2.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    dot_product / (norm1 * norm2)
}

fn calculate_total_parameters() -> usize {
    // STM weights + STM bias + NSTM weights + NSTM bias + Output weights + Output bias
    INPUT_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + INPUT_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
        + HIDDEN_SIZE * 2
        + 1
}
