use num::complex::Complex;
use ndarray::ArrayView;
use ndarray::{Dim, Ix};

use crate::{F, MC, PI};

pub struct Position {
    pub x: F,
    pub y: Option<F>,
    pub z: F,
}

#[inline(always)]
pub fn dist_img(r1: &Position, r2: &Position) -> F {
    // TODO: Check for y dimension
    ((r1.x - r2.x).powi(2) + (r2.z + r1.z).powi(2)).sqrt()
}

#[inline(always)]
pub fn dist_3_d(r_ele: &Position, x: F, y: F, z: F) -> F {
    ((x - r_ele.x).powi(2) + (y - r_ele.y.unwrap()).powi(2) + (z - r_ele.z).powi(2)).sqrt()
}

#[inline(always)]
pub fn dist_2_d(r_ele: &Position, x: F, z: F) -> F {
    ((x - r_ele.x).powi(2) + (z - r_ele.z).powi(2)).sqrt()
}

#[inline(always)]
pub fn proj_3_d(r_ele: &Position, x: F, y: F, eta: ArrayView<F, Dim<[Ix; 1]>>) -> F {
    (eta[0] - r_ele.z) - eta[1] * (x - r_ele.x) - eta[2] * (y - r_ele.y.unwrap())
}

#[inline(always)]
pub fn proj_2_d(r_ele: &Position, x: F, eta: ArrayView<F, Dim<[Ix; 1]>>) -> F {
    (eta[0] - r_ele.z) - eta[1] * (x - r_ele.x)
}

#[inline(always)]
pub fn dist_shift(d_d: F, k_a: ArrayView<F, Dim<[Ix; 1]>>) -> MC {
    k_a.map(|&k| Complex::new(0.0, -2.0 * PI * d_d * k).exp())
}

