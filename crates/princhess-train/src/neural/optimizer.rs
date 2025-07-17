use crate::neural::matrix::Matrix;
use crate::neural::vector::Vector;

#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

pub struct AdamWOptimizer {
    config: AdamWConfig,
    step: u32,
}

impl AdamWOptimizer {
    pub fn new(config: AdamWConfig) -> Self {
        Self { config, step: 0 }
    }

    pub fn step(&mut self) {
        self.step += 1;
    }

    pub fn get_step(&self) -> u32 {
        self.step
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    pub fn update_vector<const N: usize>(
        &self,
        param: &mut Vector<N>,
        grad: &Vector<N>,
        momentum: &mut Vector<N>,
        velocity: &mut Vector<N>,
        adj: f32,
    ) {
        let scaled_grad = adj * *grad;

        let bias_correction1 = 1.0 - self.config.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.config.beta2.powi(self.step as i32);

        *momentum = self.config.beta1 * *momentum + (1.0 - self.config.beta1) * scaled_grad;
        *velocity =
            self.config.beta2 * *velocity + (1.0 - self.config.beta2) * (scaled_grad * scaled_grad);

        let corrected_momentum = *momentum / bias_correction1;
        let corrected_velocity = *velocity / bias_correction2;

        let weight_decay_update = self.config.weight_decay * *param;

        *param -= self.config.learning_rate
            * (corrected_momentum / (corrected_velocity.sqrt() + self.config.epsilon)
                + weight_decay_update);
    }

    pub fn update_matrix<const R: usize, const C: usize>(
        &self,
        param: &mut Matrix<R, C>,
        grad: &Matrix<R, C>,
        momentum: &mut Matrix<R, C>,
        velocity: &mut Matrix<R, C>,
        adj: f32,
    ) {
        for i in 0..R {
            self.update_vector(
                &mut param[i],
                &grad[i],
                &mut momentum[i],
                &mut velocity[i],
                adj,
            );
        }
    }
}
