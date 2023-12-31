const E: f64 = std::f64::consts::E;

/// Activation Functions to be used in the hidden layer.
#[derive(Clone)]
pub enum ActivationFunction {
    /// Exponential Linear Units
    ///
    /// # Papers:
    ///
    /// - (2016) <https://arxiv.org/pdf/1511.07289.pdf>
    ///
    /// ![ELU](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Exponential%20Linear%20Units.png?raw=true)
    ELU,
    /// Leaky Rectified Linear Units
    ///
    /// # Papers:
    ///
    /// - (2013) <https://ai.stanford.edu/%7Eamaas/papers/relu_hybrid_icml2013_final.pdf>
    ///
    /// - (2015) <https://arxiv.org/abs/1505.00853>
    ///
    /// ![LeakyReLU](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Leaky%20Rectified%20Linear%20Units.png?raw=true)
    LeakyReLU,
    /// Identity
    ///
    /// ![Linear](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Linear.png?raw=true)
    Linear,
    /// Rectified Linear Units
    ///
    /// # Papers:
    ///
    /// - (2010) <https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf>
    ///
    /// - (2018) <https://arxiv.org/abs/1803.08375>
    ///
    /// ![ReLU](https://github.com/mbfernan/elm/blob/5102a66694bf1c8d80cff766c95e0f686424db98/src/docs/activation_functions/Rectified%20Linear%20Units.png?raw=true)
    ReLU,
    /// Sigmoidal
    ///
    /// # References:
    ///
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Sigmoid_function>
    ///
    /// ![Sigmoidal](https://github.com/mbfernan/elm/blob/7fe75cc141db217458fa75fa30b4a3252cc9acca/src/docs/activation_functions/Sigmoidal.png?raw=true)
    Sigmoidal,
    /// Step function at 0
    ///
    /// ![Step](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Step.png?raw=true)
    Step,
    /// Hyperbolic tangent
    ///
    /// # References:
    ///
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Hyperbolic_functions>
    ///
    /// ![TanH](https://github.com/mbfernan/elm/blob/e3d5484c9680773f4694a8c88f15646fab399a3d/src/docs/activation_functions/Hyperbolic%20Tangent.png?raw=true)
    TanH,
}

/// Map ActivationFunction enum variant to its function
pub fn map(activation_function: &ActivationFunction) -> fn(&mut f64) {
    match activation_function {
        ActivationFunction::ELU => elu,
        ActivationFunction::LeakyReLU => leaky_relu,
        ActivationFunction::Linear => linear,
        ActivationFunction::ReLU => relu,
        ActivationFunction::Sigmoidal => sigmoidal,
        ActivationFunction::Step => step,
        ActivationFunction::TanH => tanh,
    }
}

pub const ALPHA_ELU: f64 = 1.0;
/// [`Exponential Linear Units`]
///
/// ```
/// use elm::activation_functions::elu;
///
/// // When x >= 0.0
/// let mut x = 18.04;
/// elu(&mut x);
/// assert_eq!(x, 18.04);
///
/// // when x == 0.0
/// let mut x = 0.0;
/// elu(&mut x);
/// assert_eq!(x, 0.0);
///
/// // when x << 0.0
/// let mut x = -1000.0;
/// elu(&mut x);
/// assert_eq!(x, -1.0);
///
/// ```
///
/// [`Exponential Linear Units`]: enum.ActivationFunction.html#variant.ELU
pub fn elu(x: &mut f64) {
    if *x < 0.0 {
        *x = ALPHA_ELU * (E.powf(*x) - 1.0);
    }
}

pub const ALPHA_LEAKY_RELU: f64 = 0.1;
/// [`Leaky Rectified Linear Units`]
///
/// ```
/// use elm::activation_functions::{ALPHA_LEAKY_RELU, leaky_relu};
///
/// // When x >= 0.0
/// let mut x = 18.04;
/// leaky_relu(&mut x);
/// assert_eq!(x, 18.04);
///
/// // when x < 0.0
/// let mut x = -18.04;
/// leaky_relu(&mut x);
/// assert_eq!(x, -18.04 * ALPHA_LEAKY_RELU);
/// ```
///
/// [`Leaky Rectified Linear Units`]: enum.ActivationFunction.html#variant.LeakyReLU
pub fn leaky_relu(x: &mut f64) {
    if *x < 0.0 {
        *x *= ALPHA_LEAKY_RELU;
    }
}

/// [`Linear`]
///
/// ```
/// use elm::activation_functions::linear;
///
/// // When x == 0.0
/// let mut x = 18.04;
/// linear(&mut x);
/// assert_eq!(x, 18.04);
/// ```
///
/// [`Linear`]: enum.ActivationFunction.html#variant.Linear
pub fn linear(_x: &mut f64) {}

/// [`Rectified Linear Units`]
///
/// ```
/// use elm::activation_functions::relu;
///
/// // When x >= 0.0
/// let mut x = 18.04;
/// relu(&mut x);
/// assert_eq!(x, 18.04);
///
/// // when x < 0.0
/// let mut x = -18.04;
/// relu(&mut x);
/// assert_eq!(x, 0.0);
/// ```
///
/// [`Rectified Linear Units`]: enum.ActivationFunction.html#variant.ReLU
pub fn relu(x: &mut f64) {
    if *x < 0.0 {
        *x = 0.0;
    }
}

/// [`Sigmoidal`]
///
/// ```
/// use elm::activation_functions::sigmoidal;
///
/// // When x == 0.0
/// let mut x = 0.0;
/// sigmoidal(&mut x);
/// assert_eq!(x, 0.5);
///
/// // When x is large
/// let mut x = 1000.0;
/// sigmoidal(&mut x);
/// assert_eq!(x, 1.0);
///
/// // When x is small
/// let mut x = -1000.0;
/// sigmoidal(&mut x);
/// assert_eq!(x, 0.0);
/// ```
///
/// [`Sigmoidal`]: enum.ActivationFunction.html#variant.Sigmoidal
pub fn sigmoidal(x: &mut f64) {
    *x = 1.0 / (1.0 + E.powf(-(*x)));
}

/// [`Step`]
///
/// ```
/// use elm::activation_functions::step;
///
/// // When x == 0.0
/// let mut x = 0.0;
/// step(&mut x);
/// assert_eq!(x, 1.0);
///
/// // When x >> 0.0
/// let mut x = 10000.0;
/// step(&mut x);
/// assert_eq!(x, 1.0);
///
/// // When x << 0.0
/// let mut x = -10000.0;
/// step(&mut x);
/// assert_eq!(x, 0.0);
///
/// ```
///
/// [`Step`]: enum.ActivationFunction.html#variant.Step
pub fn step(x: &mut f64) {
    if *x < 0.0 {
        *x = 0.0;
    } else {
        *x = 1.0;
    }
}

/// [`Hyperbolic Tangent`]
///
/// ```
/// use elm::activation_functions::tanh;
///
/// // When x == 0.0
/// let mut x = 0.0;
/// tanh(&mut x);
/// assert_eq!(x, 0.0);
///
/// // When x >> 0.0
/// let mut x = 10000.0;
/// tanh(&mut x);
/// assert_eq!(x, 1.0);
///
/// // When x << 0.0
/// let mut x = -10000.0;
/// tanh(&mut x);
/// assert_eq!(x, -1.0);
///
/// ```
///
/// [`Hyperbolic Tangent`]: enum.ActivationFunction.html#variant.TanH
pub fn tanh(x: &mut f64) {
    *x = 2.0 / (1.0 + E.powf(-2.0 * (*x))) - 1.0;
}
