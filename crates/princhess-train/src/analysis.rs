use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use toml::{Table, Value};

pub struct LrAnalysisConfig<'a> {
    pub name: &'a str,
    pub input_file: &'a str,
    pub network_info: &'a str,
    pub data_positions: usize,
    pub scheduler: &'a str,
    pub total_super_batches: usize,
}

pub fn write_lr_analysis_toml(
    dir: &Path,
    config: &LrAnalysisConfig,
    grad_l1: &[f32],
    lr_full: &[f32],
) {
    if grad_l1.is_empty() || lr_full.is_empty() {
        return;
    }

    let t = grad_l1.len();
    let win = (t / 10).max(1);
    let smoothed = sliding_median(grad_l1, win);

    // Importance weights: regions with small gradient L1 norms are underexplored
    let weights: Vec<f32> = smoothed.iter().map(|&s| 1.0 / s.max(1e-9)).collect();

    // Suffix sums for the optimal forward-pass LR allocation
    let mut suffix = vec![0.0f32; t + 1];
    for i in (0..t).rev() {
        suffix[i] = suffix[i + 1] + weights[i];
    }

    // Proposed LR proportional to w[t] * remaining_weight
    let mut proposal: Vec<f32> = (0..t).map(|i| weights[i] * suffix[i + 1]).collect();

    // Normalize to match the total area under the actual LR schedule
    let proposal_sum: f32 = proposal.iter().sum();
    if proposal_sum > 0.0 {
        let actual_sum: f32 = lr_full.iter().sum();
        let scale = actual_sum / proposal_sum;
        for p in &mut proposal {
            *p *= scale;
        }
    }

    let total_super_batches = config.total_super_batches;
    let steps_per_sb = (t / total_super_batches).max(1);
    let mut schedule = Vec::new();
    for sb in 0..total_super_batches {
        let start = sb * steps_per_sb;
        let end = ((sb + 1) * steps_per_sb).min(t);
        if start >= t {
            break;
        }
        let len = (end - start) as f32;
        let actual_avg = lr_full[start..end].iter().sum::<f32>() / len;
        let proposed_avg = proposal[start..end].iter().sum::<f32>() / len;
        let grad_l1_avg = grad_l1[start..end].iter().sum::<f32>() / len;
        let smoothed_avg = smoothed[start..end].iter().sum::<f32>() / len;

        let mut entry = Table::new();
        entry.insert(
            "superbatch".into(),
            Value::Integer(i64::try_from(sb + 1).unwrap_or(i64::MAX)),
        );
        entry.insert("actual_lr".into(), actual_avg.into());
        entry.insert("proposed_lr".into(), proposed_avg.into());
        entry.insert("grad_l1".into(), grad_l1_avg.into());
        entry.insert("smoothed_grad_l1".into(), smoothed_avg.into());
        schedule.push(Value::Table(entry));
    }

    // Fine-grained warmup analysis: 10 bins each for SB1 and SB2
    const WARMUP_FINE_SBS: usize = 2;
    const WARMUP_BINS_PER_SB: usize = 10;
    let fine_steps = (steps_per_sb * WARMUP_FINE_SBS).min(t);
    let bin_size = fine_steps / (WARMUP_FINE_SBS * WARMUP_BINS_PER_SB);
    let mut warmup_analysis = Vec::new();
    if bin_size > 0 {
        for bin in 0..(WARMUP_FINE_SBS * WARMUP_BINS_PER_SB) {
            let start = bin * bin_size;
            let end = ((bin + 1) * bin_size).min(t);
            let len = (end - start) as f32;
            let step_mid = (start + end) / 2;
            let actual_avg = lr_full[start..end].iter().sum::<f32>() / len;
            let proposed_avg = proposal[start..end].iter().sum::<f32>() / len;
            let grad_l1_avg = grad_l1[start..end].iter().sum::<f32>() / len;
            let mut entry = Table::new();
            entry.insert("step_mid".into(), Value::Integer(step_mid as i64));
            entry.insert("actual_lr".into(), actual_avg.into());
            entry.insert("proposed_lr".into(), proposed_avg.into());
            entry.insert("grad_l1".into(), grad_l1_avg.into());
            warmup_analysis.push(Value::Table(entry));
        }
    }

    // Step at which proposed_lr peaks (skip step 0 — artificially high due to low initial grad norms)
    let (peak_step, &peak_proposed) = proposal[1..fine_steps]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i + 1, v))
        .unwrap_or((0, &0.0));
    let peak_actual = lr_full.get(peak_step).copied().unwrap_or(0.0);
    let mut warmup_peak = Table::new();
    warmup_peak.insert("step".into(), Value::Integer(peak_step as i64));
    warmup_peak.insert("proposed_lr".into(), peak_proposed.into());
    warmup_peak.insert("actual_lr".into(), peak_actual.into());

    let mut doc = Table::new();
    doc.insert("input_file".into(), config.input_file.to_owned().into());
    doc.insert("network_info".into(), config.network_info.to_owned().into());
    doc.insert(
        "data_positions".into(),
        Value::Integer(i64::try_from(config.data_positions).unwrap_or(i64::MAX)),
    );
    doc.insert("scheduler".into(), config.scheduler.to_owned().into());
    doc.insert("warmup_peak".into(), Value::Table(warmup_peak));
    doc.insert("warmup_analysis".into(), Value::Array(warmup_analysis));
    doc.insert("schedule".into(), Value::Array(schedule));

    let path = dir.join(format!("{}-lr-analysis.toml", config.name));
    if let Ok(mut file) = File::create(path) {
        let _ = write!(file, "{doc}");
    }

    // Raw f32 curves for detailed offline analysis: three contiguous blocks (proposed, actual, grad_l1)
    let bin_path = dir.join(format!("{}-lr-curves.bin", config.name));
    if let Ok(mut file) = File::create(bin_path) {
        let _ = file.write_all(bytemuck::cast_slice(&proposal));
        let _ = file.write_all(bytemuck::cast_slice(lr_full));
        let _ = file.write_all(bytemuck::cast_slice(grad_l1));
    }
}

#[derive(Clone, Copy)]
struct OrdF32(u32);

impl OrdF32 {
    fn new(f: f32) -> Self {
        Self(f.to_bits())
    }
    fn val(self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl PartialEq for OrdF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.val().total_cmp(&other.val())
    }
}

// Two-heap sliding median with lazy deletion. O(T log W).
fn sliding_median(data: &[f32], width: usize) -> Vec<f32> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }

    let half = width / 2;
    let w = 2 * half + 1;

    let mut padded = Vec::with_capacity(n + 2 * half);
    padded.extend(std::iter::repeat_n(data[0], half));
    padded.extend_from_slice(data);
    padded.extend(std::iter::repeat_n(*data.last().unwrap(), half));

    let mut lo: BinaryHeap<OrdF32> = BinaryHeap::new();
    let mut hi: BinaryHeap<Reverse<OrdF32>> = BinaryHeap::new();
    let mut lo_eff: i64 = 0;
    let mut hi_eff: i64 = 0;
    let mut lazy: HashMap<u32, i64> = HashMap::new();

    macro_rules! clean_lo {
        () => {
            while let Some(&top) = lo.peek() {
                if *lazy.get(&top.0).unwrap_or(&0) > 0 {
                    lo.pop();
                    *lazy.entry(top.0).or_default() -= 1;
                } else {
                    break;
                }
            }
        };
    }
    macro_rules! clean_hi {
        () => {
            while let Some(&Reverse(top)) = hi.peek() {
                if *lazy.get(&top.0).unwrap_or(&0) > 0 {
                    hi.pop();
                    *lazy.entry(top.0).or_default() -= 1;
                } else {
                    break;
                }
            }
        };
    }
    macro_rules! rebalance {
        () => {
            while lo_eff > hi_eff + 1 {
                clean_lo!();
                if let Some(top) = lo.pop() {
                    lo_eff -= 1;
                    hi.push(Reverse(top));
                    hi_eff += 1;
                }
            }
            while hi_eff > lo_eff {
                clean_hi!();
                if let Some(Reverse(top)) = hi.pop() {
                    hi_eff -= 1;
                    lo.push(top);
                    lo_eff += 1;
                }
            }
        };
    }
    macro_rules! heap_push {
        ($v:expr) => {
            clean_lo!();
            match lo.peek() {
                Some(&top) if $v > top => {
                    hi.push(Reverse($v));
                    hi_eff += 1;
                }
                _ => {
                    lo.push($v);
                    lo_eff += 1;
                }
            }
            rebalance!();
        };
    }
    macro_rules! heap_remove {
        ($v:expr) => {
            clean_lo!();
            let in_hi = matches!(lo.peek(), Some(&top) if $v > top);
            *lazy.entry($v.0).or_default() += 1;
            if in_hi { hi_eff -= 1; } else { lo_eff -= 1; }
            rebalance!();
        };
    }
    macro_rules! median {
        () => {{
            clean_lo!();
            lo.peek().map_or(0.0, |top| top.val())
        }};
    }

    for &x in &padded[..w] {
        heap_push!(OrdF32::new(x));
    }

    let mut result = Vec::with_capacity(n);
    result.push(median!());

    for i in w..padded.len() {
        heap_push!(OrdF32::new(padded[i]));
        heap_remove!(OrdF32::new(padded[i - w]));
        result.push(median!());
    }

    result
}
