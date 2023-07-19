use ndarray::{Zip, Array1, Array2, Axis, s};
use crate::{F, C, IMG};
use crate::k_axis;
use crate::t_axis;
use crate::io_experiment::Static;
use crate::surface::geometry::Surface1d;
use num::complex::Complex;
use realfft::{RealFftPlanner};

pub fn dist_img(r1: &Array1<F>, r2: &Array1<F>) -> F {
    let l = r1.len() - 1;
    let res = (r1[0] - r2[0]).powi(2) + (r2[l] + r1[l]).powi(2);
    if l == 2 {
        res += (r1[1] - r2[1]).powi(2);
    }
    l.sqrt()
}

#[inline(always)]
pub fn dist_2_d(r_ele: &Array1<F>, x: F, z: F) -> F {
    ((x - r_ele[0]).powi(2) + (z - r_ele[2]).powi(2)).sqrt()
}

#[inline(always)]
pub fn proj_2_d(r_ele: &Array1<F>, x: F, eta: ArrayView<F, Dim<[Ix; 1]>>) -> F {
    (eta[0] - r_ele[2]) - eta[1] * (x - r_ele[0])
}

#[inline(always)]
pub fn scale_and_shift(amp: F, d_d: F, k_a: ArrayView<F, Dim<[Ix; 1]>>) -> MC {
    k_a.map(|&k| amp * Complex::new(0.0, -2.0 * PI * d_d * k).exp())
}

pub fn dist_bound(stat: &Static, eta_1D: &Array2<F>) -> Array1<F>{
    let r_max = dist_img(&stat.r_src, &stat.r_rcr) + stat.duration * stat.c;

    let ncols = 3;

    let mut data = Vec::new();
    let mut nrows = 0;

    Zip::from(&surface.x_axis()).and(eta_1D.lanes(Axis(1))).for_each(|&x, z| {

        let d_s = dist_2_d(&stat.r_src, x, z[0]);
        let d_r = dist_2_d(&stat.r_rcr, x, z[0]);
        let d = d_s + d_r;

        if d <= r_max {
            let p = proj_2_d(&stat.r_src, x, z);
            let row = vec![d_s, d_r, p];
            data.extend_from_slice(&row);
            nrows += 1;
        }
    });
    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}

pub fn ka_sum_1d(stat: &Static, surface: &Surface1d) -> Array1<F>{

    let params = ier_param_2D(stat, surface);
    let dr = stat.c / stat.pulse.fs;
    let d_0 = stat.tau_0 * stat.c;

    // integer sample result
    let t_a = t_axis(stat.pulse.fs, stat.duration, stat.tau_0);
    let n_pulse = stat.pulse.signal.len();
    let n_pulse_ft = n_pulse / 2 + 1;
    let k_a_pulse = k_axis(stat.pulse.fs, (n_pulse as F) / stat.pulse.fs, stat.c);

    // discretized acculation step
    let mut igral: Array2<C> = Array2::zeros((t_a.len(), n_pulse_ft));

    Zip::from(params.rows()).for_each(|p| {
        let scale = Complex::new(p[2] / (p[0].powi(2) * p[1]), 0.0) ;
        let d_rel = p[0] + p[1] - d_0;
        let d_samps = d_rel / dr;
        let ind = d_samps as usize;
        if ind < t_a.len() - 1 {
            let mut ft = igral.index_axis_mut(Axis(0), ind);
            let infin: Array1<C> = k_a_pulse.map(|k| scale * (-IMG * k * d_rel).exp());
            ft += &infin;
        }
    });

    let mut result: Array1<F> = Array1::zeros(t_a.raw_dim()[0] + n_pulse);

    igral.rows_mut().into_iter().enumerate().for_each(|(i, mut e)| {
        let mut outdata = c2r.make_output_vec();
        let mut slice = result.slice_mut(s![i .. i + stat.pulse.signal.raw_dim()[0]]);
        e *= &pulse_ft;
        c2r.process(&mut e.to_vec(), &mut outdata).unwrap();
        let shift_pulse = Array1::from_vec(outdata);
        slice += &shift_pulse;
    });
    result
}
