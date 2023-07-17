use ndarray::prelude::{Array1, Array2};
use std::f64::consts::PI as FullPI;
use num::complex::Complex;

pub mod io_experiment;
pub mod greens;
pub mod ka_sum;
pub mod xmission;
pub mod surface;

#[cfg(feature = "bigger")] pub type F = f64;
#[cfg(not(feature = "bigger"))] pub type F = f32;

pub type C = Complex<F>;
pub type M = Array1<F>;
pub type MM = Array2<F>;
pub type MC = Array1<Complex<F>>;

pub const IMG: C = Complex::new(0.0, 1.0);
pub const PI: F = FullPI as F;

pub fn t_axis(fs: F, duration: F, tau_0: F) -> M {
    let dt = 1.0 / fs;
    let n = axis_size(fs, duration);

    Array1::range(0.0, n as F, 1.0) * dt + tau_0
}

pub fn f_axis(fs: F, duration: F) -> M {
    let n = axis_size(fs, duration);
    let df = n / fs;
    Array1::range(0.0, (n / 2.0 + 1.0) as F, 1.0) * df / n
}

pub fn k_axis(fs: F, duration: F, c: F) -> M {
    let f_a = f_axis(fs, duration);
    2.0 * PI * f_a / c
}

fn axis_size(fs: F, duration: F) -> F {
    let mut n: F = (duration * fs).ceil();
    if (n % 2.0) > 0.5 {n += 1.0};
    n
}
