use ndarray::Zip;
use num::Complex;
use kasea::{F, M, MM, MC, IMG, PI2};

#[inline(always)]
fn dist_2D(r_ele: &M, x: F, z: F) -> F {
    ((r_ele[0] - x).powi(2) + (r_ele[2] - z).powi(2)).sqrt()
}

#[inline(always)]
fn dist_3D(r_ele: &M, x: F, z: F) -> F {
    // not implimented
    ((r_ele[0] - x).powi(2) + (r_ele[2] - z).powi(2)).sqrt()
}

#[inline(always)]
fn proj_2D(r_ele: &M, dist_ele: F, x: F, z: F, z_x: F) -> F {
    ((z - &r_ele[2]) - z_x * (x - &r_ele[0])) / dist_ele
}

#[inline(always)]
fn proj_3D(r_ele: &M, dist_ele: F, x: F, z: F, z_x: F) -> F {
    // not implimented
    ((z - &r_ele[2]) - z_x * (x - &r_ele[0])) / dist_ele
}

pub enum Surface {
    M,
    MM,
}

pub struct Specs {
    pub r_src: M,
    pub r_rcr: M,
    pub r_max: F,
    pub x_a: M,
    pub y_a: Option<M>,
    pub k_a: M,
    pub eta: Surface,
    pub eta_x: Surface,
    pub eta_y: Option<Surface>,
}

pub fn ka_sum_1d(spec: Specs) -> MC {

    let d0 = MC::zeros(spec.k_a.raw_dim()[0]);

    Zip::from(&spec.x_a).and(&spec.eta).and(&spec.eta_x).fold(d0, |acc, &x, &z, &z_x| {

        let d_s = dist_2D(&spec.r_src, x, z);
        let d_r = dist_2D(&spec.r_rcr, x, z);
        let d = d_s + d_r;

        if d <= spec.r_max {
            let p = proj_2D(&spec.r_src, d_s, x, z, z_x);
            acc + spec.k_a.map(|&k| -IMG * k * p * Complex::new(0.0, -k * d).exp()
                                    / (16.0 * PI2 * d_s * d_r))
        } else {
            acc
        }
    })
}
