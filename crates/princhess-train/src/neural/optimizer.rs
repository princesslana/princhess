use crate::neural::lr_scheduler::LRScheduler;
use crate::neural::matrix::Matrix;
use crate::neural::vector::Vector;

pub struct AdamWOptimizer<S: LRScheduler> {
    scheduler: S,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: u32,
    bias_correction1: f32,
    bias_correction2: f32,
}

impl<S: LRScheduler> AdamWOptimizer<S> {
    pub fn with_scheduler(scheduler: S) -> Self {
        Self {
            scheduler,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            step: 0,
            bias_correction1: 1.0,
            bias_correction2: 1.0,
        }
    }

    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn step(&mut self) {
        self.step += 1;
        self.bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        self.bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);
    }

    pub fn get_step(&self) -> u32 {
        self.step
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.scheduler.get_lr(self.step)
    }

    pub fn update_vector<const N: usize>(
        &self,
        param: &mut Vector<N>,
        grad: &Vector<N>,
        momentum: &mut Vector<N>,
        velocity: &mut Vector<N>,
    ) {
        *momentum = self.beta1 * *momentum + (1.0 - self.beta1) * *grad;
        *velocity = self.beta2 * *velocity + (1.0 - self.beta2) * (*grad * *grad);

        let corrected_momentum = *momentum / self.bias_correction1;
        let corrected_velocity = *velocity / self.bias_correction2;

        let weight_decay_update = self.weight_decay * *param;

        let learning_rate = self.scheduler.get_lr(self.step);
        *param -= learning_rate
            * (corrected_momentum / (corrected_velocity.sqrt() + self.epsilon)
                + weight_decay_update);
    }

    pub fn update_matrix<const R: usize, const C: usize>(
        &self,
        param: &mut Matrix<R, C>,
        grad: &Matrix<R, C>,
        momentum: &mut Matrix<R, C>,
        velocity: &mut Matrix<R, C>,
    ) {
        for i in 0..R {
            self.update_vector(&mut param[i], &grad[i], &mut momentum[i], &mut velocity[i]);
        }
    }
}
