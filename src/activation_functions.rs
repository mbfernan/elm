const E: f64 = std::f64::consts::E;

/// Activation Functions to be used in the hidden layer.
#[derive(Clone)]
pub enum ActivationFunction {
    /// Leaky Rectified Linear Units
    ///
    /// # Papers:
    ///
    /// - (2013) [<https://ai.stanford.edu/%7Eamaas/papers/relu_hybrid_icml2013_final.pdf>]
    ///
    /// - (2015) [<https://arxiv.org/abs/1505.00853>]
    LeakyRelu,
    /// Sigmoidal
    ///
    /// # References:
    ///
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Sigmoid_function>
    Sigmoidal,
    /// Linear
    ///
    /// Not widely used in the ELM environment.
    Linear,
    /// Step function at 0
    Step,
    /// Hyperbolic tangent
    ///
    ///  # References:
    ///
    /// - Wikipedia: <https://en.wikipedia.org/wiki/Hyperbolic_functions>
    TanH,
}

/// Map ActivationFunction enum variant to its function
pub fn map(activation_function: &ActivationFunction) -> fn(&mut f64) {
    match activation_function {
        ActivationFunction::LeakyRelu => leaky_relu,
        ActivationFunction::Sigmoidal => sigmoidal,
        ActivationFunction::Linear => linear,
        ActivationFunction::Step => step,
        ActivationFunction::TanH => tanh,
    }
}

/// [`Leaky Rectified Linear Units`]
///
/// ```
/// use elm::activation_functions::leaky_relu;
///
/// // When x >= 0.0
/// let mut x = 18.04;
/// leaky_relu(&mut x);
/// assert_eq!(x, 18.04);
///
/// // when x < 0.0
/// let mut x = -18.04;
/// leaky_relu(&mut x);
/// assert_eq!(x, -18.04 * 0.01);
/// ```
///
/// [`Leaky Rectified Linear Units`]: enum.ActivationFunction.html#variant.leakyrelu
pub fn leaky_relu(x: &mut f64) {
    if *x < 0.0 {
        *x *= 0.01;
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
/// [`Sigmoidal`]: enum.ActivationFunction.html#variant.sigmoidal
pub fn sigmoidal(x: &mut f64) {
    *x = 1.0 / (1.0 + E.powf(-(*x)));
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
/// [`Linear`]: enum.ActivationFunction.html#variant.linear
pub fn linear(_x: &mut f64) {}

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
