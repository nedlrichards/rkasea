use ndarray::prelude::{Array1, Array2};
use std::f64::consts::PI as FullPI;
use num::complex::Complex;
mod geometry;
mod greens;

#[cfg(feature = "bigger")] pub type F = f64;
#[cfg(not(feature = "bigger"))] pub type F = f32;

pub use geometry::*;
pub use greens::*;

pub type C = Complex<F>;
pub type M = Array1<F>;
pub type MM = Array2<F>;
pub type MC = Array1<Complex<F>>;

pub const IMG: C = Complex::new(0.0, 1.0);
pub const PI: F = FullPI as F;
pub const PI2: F = PI * PI;


