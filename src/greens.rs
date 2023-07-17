use num::complex::Complex;
use ndarray::{Dim, Ix, Array1, ArrayView};

use crate::{F, MC, PI};

#[inline(always)]
pub fn dist_img(r1: &Array1<F>, r2: &Array1<F>) -> F {
    ((r1[0] - r2[0]).powi(2) + (r1[1] - r2[1]).powi(2) + (r2[2] + r1[2]).powi(2)).sqrt()
}

#[inline(always)]
pub fn dist_3_d(r_ele: &Array1<F>, x: F, y: F, z: F) -> F {
    ((x - r_ele[0]).powi(2) + (y - r_ele[1]).powi(2) + (z - r_ele[2]).powi(2)).sqrt()
}

#[inline(always)]
pub fn dist_2_d(r_ele: &Array1<F>, x: F, z: F) -> F {
    ((x - r_ele[0]).powi(2) + (z - r_ele[2]).powi(2)).sqrt()
}

#[inline(always)]
pub fn proj_3_d(r_ele: &Array1<F>, x: F, y: F, eta: ArrayView<F, Dim<[Ix; 1]>>) -> F {
    (eta[0] - r_ele[2]) - eta[1] * (x - r_ele[0]) - eta[2] * (y - r_ele[1])
}

#[inline(always)]
pub fn proj_2_d(r_ele: &Array1<F>, x: F, eta: ArrayView<F, Dim<[Ix; 1]>>) -> F {
    (eta[0] - r_ele[2]) - eta[1] * (x - r_ele[0])
}

#[inline(always)]
pub fn dist_shift(d_d: F, k_a: ArrayView<F, Dim<[Ix; 1]>>) -> MC {
    k_a.map(|&k| Complex::new(0.0, -2.0 * PI * d_d * k).exp())
}

