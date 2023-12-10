use std::time::Duration;


/// Construct wrapper for saving all relevant ELM training metrics.
///
/// If **pseudo inverse** could not be calculated, some training metrics will be `None`.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct ELMTrainingMetrics {
    pub training_mse: Option<f64>,
    pub training_duration: Option<Duration>,
}

impl ELMTrainingMetrics {
    pub fn display(&self) {
        println!("\nELM training metrics:");
        match &self.training_mse {
            Some(mse) => println!("\tMSE: {}", mse),
            None => println!("\tMSE: -"),
        }
        match &self.training_duration {
            Some(dur) => println!("\tTraining duration: {} µs\n", dur.as_micros()),
            None => println!("\tTraining duration: -\n"),
        }
    }
}