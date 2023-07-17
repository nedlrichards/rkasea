use ndarray::{Zip, Array1, Array2, Axis, s};
use crate::{F, C, IMG};
use crate::k_axis;
use crate::t_axis;
use crate::io_experiment::Static;
use crate::surface::geometry::Surface1d;
use crate::greens::{dist_img, proj_2_d, dist_2_d};
use num::complex::Complex;
use realfft::{RealFftPlanner};

pub fn ka_sum_1d(stat: &Static, surface: &Surface1d) -> Array1<F>{

    let params = ier_param_2D(stat, surface);
    let dr = stat.c / stat.pulse.fs;
    let d_0 = stat.tau_0 * stat.c;

    // integer sample result
    let t_a = t_axis(stat.pulse.fs, stat.duration, stat.tau_0);
    let n_pulse = stat.pulse.signal.len();
    let n_pulse_ft = n_pulse / 2 + 1;
    let k_a_pulse = k_axis(stat.pulse.fs, (n_pulse as F) / stat.pulse.fs, stat.c);

    //setup t -> k fft
    let mut planner = RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(n_pulse);
    let mut pulse_ft = r2c.make_output_vec();
    let mut indata = stat.pulse.signal.clone().to_vec();
    r2c.process(&mut indata, &mut pulse_ft).unwrap();
    let mut pulse_ft = Array1::from_vec(pulse_ft);
    let np = &pulse_ft.len();
    pulse_ft[np - 1] = Complex::new(0.0, 0.0);

    //setup k -> t fft
    let c2r = planner.plan_fft_inverse(n_pulse);

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


fn ier_param_2D(stat: &Static, surface: &Surface1d) -> Array2<F> {
    // transform surface and element positions into range and amplitude vector

    let r_max = dist_img(&stat.r_src, &stat.r_rcr) + stat.duration * stat.c;

    let ncols = 3;

    let mut data = Vec::new();
    let mut nrows = 0;

    Zip::from(&surface.x_axis()).and(surface.eta().lanes(Axis(1))).for_each(|&x, z| {

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
