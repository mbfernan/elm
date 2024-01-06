use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::BufReader, path::Path};

use crate::{
    activation_functions::{map, ActivationFunction},
    flatten_matrix, ELM,
};

#[derive(Deserialize, Serialize)]
struct IOELM {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    activation_function: ActivationFunction,
    weights: Vec<f64>,
    biases: Vec<f64>,
    beta: Vec<f64>,
    epsilon: f64,
}

impl ELM {
    /// Serializes ELM into IOELM (intermediate format) and saves it into JSON a file.
    pub fn to_json(&self, path: &Path) -> Result<(), Error> {
        let file: File = File::create(path)?;
        serde_json::to_writer(&file, &IOELM::from(self))?;
        Ok(())
    }

    /// Deserializes IOELM (intermediate format) into ELM from a given JSON file.
    pub fn from_json(path: &Path) -> Result<ELM, Error> {
        let file: File = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let io_elm: IOELM = serde_json::from_reader(buf_reader)?;
        Ok(ELM::from(io_elm))
    }
}

impl From<&ELM> for IOELM {
    fn from(elm: &ELM) -> Self {
        IOELM {
            input_size: elm.input_size,
            hidden_size: elm.hidden_size,
            output_size: elm.output_size,
            activation_function: elm.activation_function.clone(),
            weights: flatten_matrix(&elm.weights),
            biases: flatten_matrix(&elm.biases),
            beta: flatten_matrix(&elm.beta),
            epsilon: elm.epsilon,
        }
    }
}

impl From<IOELM> for ELM {
    fn from(io_elm: IOELM) -> Self {
        ELM {
            input_size: io_elm.input_size,
            hidden_size: io_elm.hidden_size,
            output_size: io_elm.output_size,
            activation_function_fn: map(&io_elm.activation_function),
            activation_function: io_elm.activation_function,
            weights: DMatrix::from_vec(
                io_elm.hidden_size,
                io_elm.input_size,
                io_elm.weights.to_vec(),
            ),
            biases: DMatrix::from_vec(1, io_elm.hidden_size, io_elm.biases.to_vec()),
            beta: DMatrix::from_vec(io_elm.hidden_size, io_elm.output_size, io_elm.beta.to_vec()),
            epsilon: io_elm.epsilon,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    /// Adapter from [`serde_json::Error`]
    JSONError(serde_json::Error),
    /// Adapter from [`std::io::Error`]
    FileError(std::io::Error),
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::JSONError(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Error::FileError(error)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::activation_functions::ActivationFunction;
    use crate::{
        loss::Loss,
        {Epsilon, ELM},
    };

    #[test]
    fn test_io() {
        const TOLERANCE: f64 = 0.000001;
        // Test saving and loading an ELM
        let path = Path::new(&"./test_elm.json");

        // Instantiate an ELM
        let mut elm_to_save = ELM::new(2, 4, 1, ActivationFunction::LeakyReLU, Epsilon::Default);

        // Train
        let inputs = vec![
            vec![0.1, 6.7],
            vec![2.1, 3.2],
            vec![0.3, 4.0],
            vec![1.0, 5.2],
            vec![4.0, 7.6],
            vec![5.0, 10.0],
            vec![0.0, 0.0],
        ];
        let targets = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
            vec![7.0],
        ];
        elm_to_save.train(&inputs, &targets);

        // Predict
        let prediction_base = elm_to_save.predict(&inputs);

        // Save
        elm_to_save.to_json(path).unwrap();

        // Load
        let loaded_elm = ELM::from_json(path).unwrap();

        // Predict with the loaded ELM
        let prediction_to_compare = loaded_elm.predict(&inputs);

        // Create a Custom Loss function that will return the maximum difference
        fn max_abs_difference(base: &[f64], to_compare: &[f64]) -> f64 {
            base.iter().zip(to_compare.iter()).fold(0.0, |acc, (a, b)| {
                let diff = (a - b).abs();
                if diff > acc {
                    diff
                } else {
                    acc
                }
            })
        }

        assert!(
            Loss::Custom(max_abs_difference).calculate(&prediction_base, &prediction_to_compare)
                < TOLERANCE
        );

        // Deleting test file
        std::fs::remove_file(path).unwrap();
    }
}
