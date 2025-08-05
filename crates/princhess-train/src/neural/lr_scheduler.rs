pub trait LRScheduler {
    fn get_lr(&self, step: u32) -> f32;
}

#[derive(Debug, Clone)]
pub struct StepLRScheduler {
    initial_lr: f32,
    drop_factor: f32,
    drop_interval: u32,
}

impl StepLRScheduler {
    pub fn new(initial_lr: f32, drop_factor: f32, drop_at_fraction: f32, total_steps: u32) -> Self {
        let drop_interval = (total_steps as f32 * drop_at_fraction).ceil() as u32;
        Self {
            initial_lr,
            drop_factor,
            drop_interval,
        }
    }
}

impl LRScheduler for StepLRScheduler {
    fn get_lr(&self, step: u32) -> f32 {
        let num_drops = step / self.drop_interval;
        self.initial_lr * self.drop_factor.powi(num_drops as i32)
    }
}

#[derive(Debug, Clone)]
pub struct ConstantLRScheduler {
    lr: f32,
}

impl ConstantLRScheduler {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LRScheduler for ConstantLRScheduler {
    fn get_lr(&self, _step: u32) -> f32 {
        self.lr
    }
}
