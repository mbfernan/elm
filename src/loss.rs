use crate::{flatten_matrix, ToMatrix};

#[derive(Clone)]
pub enum Loss {
    /// Mean Squared Error
    ///
    /// # References
    ///
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Mean_squared_error>
    MSE,
    /// Custom Loss function
    ///
    /// # Usage
    ///
    /// ```
    /// use elm::loss::Loss;
    /// use nalgebra::DMatrix;
    ///
    /// fn custom_loss_function(predictions: &[f64], targets: &[f64]) -> f64 {
    ///     predictions
    ///         .iter()
    ///         .zip(targets.iter())
    ///         .fold(0.0, |acc, (output, target)| acc + (output - target).abs())
    /// }
    ///
    /// let predictions = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
    /// let targets = vec![vec![1.0, 1.0], vec![1.0, 0.0]];
    ///
    /// let loss = Loss::Custom(custom_loss_function).calculate(&predictions, &targets);
    /// assert_eq!(loss, 1.0);
    ///
    /// ```
    Custom(fn(&[f64], &[f64]) -> f64),
}

impl Loss {
    /// Calculate loss between predicted and target values.
    ///
    /// # Usage
    ///
    /// ```
    /// use elm::loss::Loss;
    /// use nalgebra::DMatrix;
    ///
    /// // With Vec<Vec<f64>>
    /// let predictions = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
    /// let targets = vec![vec![1.0, 1.0], vec![1.0, 0.0]];
    ///
    /// let loss = Loss::MSE.calculate(&predictions, &targets);
    /// assert_eq!(loss, 0.25);
    ///
    /// // With DMatrix<f64>
    /// let predictions: DMatrix<f64> = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 1.0, 0.0]);
    /// let targets: DMatrix<f64> = DMatrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 0.0]);
    ///
    /// let loss = Loss::MSE.calculate(&predictions, &targets);
    /// assert_eq!(loss, 0.25);
    ///
    /// ```
    pub fn calculate<T: ToMatrix>(&mut self, predictions: &T, targets: &T) -> f64 {
        let flattened_predictions = flatten_matrix(&predictions.to_matrix());
        let flattened_targets = flatten_matrix(&targets.to_matrix());

        if flattened_predictions.len() != flattened_targets.len() {
            panic!("Predictions and targets length mismatch.");
        }

        match self {
            Loss::MSE => mse(&flattened_predictions, &flattened_targets),
            Loss::Custom(function) => function(&flattened_predictions, &flattened_targets),
        }
    }
}

fn mse(predictions: &[f64], targets: &[f64]) -> f64 {
    predictions
        .iter()
        .zip(targets.iter())
        .fold(0.0, |acc, (prediction, target)| {
            acc + (prediction - target).powi(2)
        })
        / predictions.len() as f64
}
