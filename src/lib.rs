//! Extreme Learning Machine (ELM) crate. A minimalistic and flexible crate that can be used to train ELMs, a type of
//! Neural Networks. Currently supports a single hidden layer and regression tasks.
//!
//! References:
//!
//! - Original paper: <https://ieeexplore.ieee.org/document/1380068>
//!
//! - Wikipedia: <https://en.wikipedia.org/wiki/Extreme_learning_machine>


/// Provides Activation Functions utilities.
pub mod activation_functions;
/// Provides Training Metrics utilities.
pub mod training_metrics;

use nalgebra::DMatrix;
use rand::{
    distributions::{Distribution, Uniform},
    rngs::ThreadRng,
};
use std::time::Instant;

use crate::activation_functions::ActivationFunction;
use crate::training_metrics::ELMTrainingMetrics;

/// Extreme Learning Machine (ELM) base struct.
pub struct ELM {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    activation_function_fn: fn(&mut f64),
    activation_function: ActivationFunction,
    weights: DMatrix<f64>,
    biases: DMatrix<f64>,
    hidden: DMatrix<f64>,
    beta: Option<DMatrix<f64>>,
    training_metrics: ELMTrainingMetrics,
    epsilon: Epsilon,
    verbose: Verbose,
}

impl ELM {
    /// Constructs an ELM Neural Network based on the specified architecture.
    ///
    /// ```input_size``` refers to the number of inputs for each data point, i.e. features.
    ///
    /// ```hidden_size``` refers to the number of nodes in the hidden layer.
    ///
    /// ```output_size``` refers to the number of outputs in the output layer.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        activation_function: ActivationFunction,
        epsilon: Epsilon,
        verbose: Verbose,
    ) -> Self {
        let mut rng: ThreadRng = rand::thread_rng();
        let weights_distribution: Uniform<f64> = Uniform::from(-0.5..=0.5);
        let biases_distribution: Uniform<f64> = Uniform::from(0.0..=1.0);

        Self {
            input_size,
            hidden_size,
            output_size,
            activation_function_fn: activation_functions::map(&activation_function),
            activation_function,
            weights: DMatrix::from_fn(hidden_size, input_size, |_, _| {
                weights_distribution.sample(&mut rng)
            }),
            biases: DMatrix::from_fn(1, hidden_size, |_, _| biases_distribution.sample(&mut rng)),
            hidden: DMatrix::zeros(hidden_size, 1),
            beta: None,
            training_metrics: ELMTrainingMetrics::default(),
            epsilon,
            verbose,
        }
    }

    fn pass_to_hidden<T: ToMatrix>(&self, inputs: &T) -> DMatrix<f64> {
        let inputs = inputs.to_matrix();
        let mut hidden = inputs * self.weights.transpose();
        hidden
            .row_iter_mut()
            .for_each(|mut row| row += &self.biases);
        hidden.apply(|x| (self.activation_function_fn)(x));
        hidden
    }

    /// Train ELM to predict the targets based on inputs.
    ///
    /// **inputs** shape: (n_data_points x input_size)
    ///
    /// **targets** shape: (n_data_points x output_size)
    ///
    /// # Data types:
    ///
    /// This function accepts inputs and targets as `Vec<Vec<f64>>` or `nalgebra::DMatrix<f64>`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use elm::{ELM, Epsilon, Verbose};
    /// use elm::activation_functions::ActivationFunction;
    ///
    /// let mut elm = ELM::new(
    ///     2,
    ///     4,
    ///     2,
    ///     ActivationFunction::LeakyReLU,
    ///     Epsilon::Default,
    ///     Verbose::Quiet,
    /// );
    ///
    /// // Each row is a data point. Note input size = 2
    /// let inputs: Vec<Vec<f64>> = vec![
    ///     vec![1.0, 0.0],
    ///     vec![1.0, 0.0],
    ///     vec![1.0, 0.0],
    ///     vec![0.0, 1.0],
    ///     vec![0.0, 1.0],
    ///     vec![1.0, 1.0],
    ///     vec![0.0, 0.0],
    /// ];
    ///
    /// // Each row is a data point. Note output size = 2
    /// let targets: Vec<Vec<f64>> = vec![
    ///     vec![1.0, 1.0],
    ///     vec![1.0, 1.5],
    ///     vec![1.0, 1.5],
    ///     vec![1.0, 0.0],
    ///     vec![1.0, 0.2],
    ///     vec![0.0, 2.0],
    ///     vec![0.0, 0.0],
    /// ];
    ///
    /// elm.train(&inputs, &targets);
    /// ```
    ///
    /// # Panics:
    ///
    /// Panics if inputs and targets have different number of data points.
    ///
    /// # Performance:
    ///
    /// If failed to calculate **pseudo inverse**, Beta will be set to `None` and no training metrics will be available.
    pub fn train<T: ToMatrix>(&mut self, inputs: &T, targets: &T) {
        let timer = Instant::now();
        self.hidden = self.pass_to_hidden(inputs);
        let moore_penrose =
            (self.hidden.transpose() * &self.hidden).pseudo_inverse(self.epsilon.get());

        let targets = targets.to_matrix();
        match moore_penrose {
            Ok(mp) => {
                self.beta = Some((mp * self.hidden.transpose()) * &targets);
                self.training_metrics = ELMTrainingMetrics {
                    training_mse: self.calculate_mse(&targets),
                    training_duration: Some(timer.elapsed()),
                };
            }
            Err(_) => {
                if self.verbose == Verbose::Full || self.verbose == Verbose::Warnings {
                    println!("WARNING: Could not calculate Pseudo Inverse.");
                }
                self.beta = None;
                self.training_metrics = ELMTrainingMetrics::default();
            }
        };

        if self.verbose == Verbose::Full || self.verbose == Verbose::TrainingMetrics {
            self.training_metrics.display();
        }
    }

    fn calculate_mse<T: ToMatrix>(&self, targets: &T) -> Option<f64> {
        let targets = targets.to_matrix();
        let flattened_targets = flatten_matrix(&targets);
        let flattened_outputs = match self.predict(&targets) {
            Some(outputs) => flatten_matrix(&outputs),
            None => return None,
        };
        if flattened_outputs.len() != flattened_targets.len() {
            if self.verbose == Verbose::Full || self.verbose == Verbose::Warnings {
                println!("MSE WARNING: target lenght different the output length.")
            }
            return None;
        }

        // Mean Squared Error
        let mse = flattened_outputs
            .iter()
            .zip(flattened_targets.iter())
            .fold(0.0, |acc, (output, target)| acc + (output - target).powi(2))
            / flattened_outputs.len() as f64;

        Some(mse)
    }

    /// Forward pass on ELM, used to predict values based on inputs provided and once the ELM has already being
    /// [`trained`].
    ///
    /// # Data types:
    ///
    /// This function accepts inputs as `Vec<Vec<f64>>` or `nalgebra::DMatrix<f64>`.
    /// Outputs will have the same type as the inputs.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use elm::{ELM, Epsilon, Verbose};
    /// use elm::activation_functions::ActivationFunction;
    ///
    /// let mut elm = ELM::new(2, 4, 2, ActivationFunction::LeakyReLU, Epsilon::Default, Verbose::Quiet);
    /// let inputs: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
    /// let targets: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![1.0, 1.5]];
    /// elm.train(&inputs, &targets);
    ///
    /// let new_inputs: Vec<Vec<f64>> = vec![vec![1.0, 4.0], vec![1.3, 0.6]];
    /// let prediction = elm.predict(&new_inputs);   // Type: Vec<Vec<f64>>
    /// ```
    ///
    /// [`trained`]: struct.ELM.html#method.train
    pub fn predict<T: ToMatrix + FromMatrix>(
        &self,
        inputs: &T,
    ) -> Option<<T as FromMatrix>::Output> {
        let hidden = self.pass_to_hidden(inputs);
        match &self.beta {
            Some(beta) => {
                let res = hidden * beta;
                let to_t = <T as FromMatrix>::from_matrix(res);
                Some(to_t)
            }
            None => None,
        }
    }

    /// Retrieves ELM input layer size.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Retrieves ELM hidden layer size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Retrieves ELM output layer size.
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Retrieves ELM activation function.
    pub fn activation_function(&self) -> ActivationFunction {
        self.activation_function.clone()
    }

    /// Retrieves training metrics.
    pub fn training_metrics(&self) -> ELMTrainingMetrics {
        self.training_metrics.clone()
    }

    /// Displayes training metrics.
    pub fn display_training_metrics(&self) {
        self.training_metrics.display();
    }
}

/// Implemented to allow flexible usage. Transforms `Vec<Vec<f64>>` or `DMatrix<f64>` into `DMatrix<f64>`.
pub trait ToMatrix {
    fn to_matrix(&self) -> DMatrix<f64>;
}

impl ToMatrix for Vec<Vec<f64>> {
    fn to_matrix(&self) -> DMatrix<f64> {
        let num_columns = self.first().map_or(0, |first_row| first_row.len());
        DMatrix::from_rows(
            &self
                .iter()
                .filter(|row| row.len() == num_columns)
                .map(|row| row.clone().into())
                .collect::<Vec<_>>(),
        )
    }
}

impl ToMatrix for DMatrix<f64> {
    fn to_matrix(&self) -> DMatrix<f64> {
        self.clone()
    }
}

/// Implemented to allow flexible usage. Transforms `DMatrix<f64>` into `Vec<Vec<f64>>` or `DMatrix<d64>`.
pub trait FromMatrix {
    type Output;

    fn from_matrix(matrix: DMatrix<f64>) -> Self::Output;
}

impl FromMatrix for DMatrix<f64> {
    type Output = DMatrix<f64>;

    fn from_matrix(matrix: DMatrix<f64>) -> Self::Output {
        matrix
    }
}

impl FromMatrix for Vec<Vec<f64>> {
    type Output = Vec<Vec<f64>>;

    fn from_matrix(matrix: DMatrix<f64>) -> Self::Output {
        matrix
            .row_iter()
            .map(|row| row.iter().cloned().collect())
            .collect()
    }
}

/// Flatten a DMatrix into a 1D vector considering columns as the primary dimension.
///
/// ```
/// use elm::{flatten_matrix, ToMatrix};
///
/// let a: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
/// assert_eq!(flatten_matrix(&a.to_matrix()), vec![0.0, 2.0, 1.0, 3.0]);
/// ```
pub fn flatten_matrix(matrix: &DMatrix<f64>) -> Vec<f64> {
    matrix.iter().cloned().collect()
}

const EPSILON: f64 = 0.0001;
/// All singular values below Epsilon are considered equal to 0.
pub enum Epsilon {
    /// Default value for Epsilon: 0.0001
    Default,
    /// User-specified value for Epsilon
    Custom(f64),
}

impl Epsilon {
    fn get(&self) -> f64 {
        match self {
            Epsilon::Default => EPSILON,
            Epsilon::Custom(eps) => *eps,
        }
    }
}

/// Used for defining verbosity.
#[derive(PartialEq)]
pub enum Verbose {
    /// No print statements will be displayed.
    Quiet,
    /// Only display warnings.
    Warnings,
    /// Only display training metrics after [`training`] is completed.
    ///
    /// [`training`]: struct.ELM.html#method.train
    TrainingMetrics,
    /// Display all print statements.
    Full,
}
