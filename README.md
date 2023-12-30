# Extreme Learning Machines (ELMs)

Extreme Learning Machine (ELM) crate. A minimalistic and flexible crate that can be used to train ELMs,
a type of Neural Networks. Currently supports a single hidden layer and regression tasks.

References:

- Original paper: <https://ieeexplore.ieee.org/document/1380068>

- Wikipedia: <https://en.wikipedia.org/wiki/Extreme_learning_machine>


## Basic usage

```RUST
use elm::{ELM, Epsilon};
use elm::activation_functions::ActivationFunction;

let mut elm = ELM::new(2, 4, 2, ActivationFunction::LeakyReLU, Epsilon::Default);
let inputs: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
let targets: Vec<Vec<f64>> = vec![vec![1.0, 1.0], vec![1.0, 1.5]];
elm.train(&inputs, &targets);

let new_inputs: Vec<Vec<f64>> = vec![vec![1.0, 4.0], vec![1.3, 0.6]];
let prediction = elm.predict(&new_inputs);
```

## Activation functions

![ELU](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Exponential%20Linear%20Units.png)

![LeakyReLU](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Leaky%20Rectified%20Linear%20Units.png)

![Linear](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Linear.png)

![ReLU](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Rectified%20Linear%20Units.png)

![Sigmoidal](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Rectified%20Linear%20Units.png)

![Step](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Step.png)

![TanH](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Hyperbolic%20Tangent.png)

## Road map

- [ ] Export module: to save and load previously trained models
- [ ] Loss trait to allow custom implementations