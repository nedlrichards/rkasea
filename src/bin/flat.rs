use ndarray::prelude::*;
use rkasea::{F, M};

use rkasea::geometry::Position;
use rkasea::xmitt::nuttall_pulse;
use rkasea::bound_axes::{bound_axes_x, bound_axes_y};
use rkasea::ka_sum;


fn main() {

    let c = 1500.0;
    let decimation = 8;
    let fc = 3.5e3;
    let fs = 2.0 * (fc * 1.6);
    let dx = c / (fs * (decimation as F));

    let c = 1500.0;
    let tau_buf = 0.2e-3;
    let duration = 10e-3;

    let r_src = Position{
        x:0.0,
        y:None,
        z:-5.0,
    };

    let r_rcr = Position{
        x:10.0,
        y:None,
        z:-10.0,
    };

    let d_img = ((r_rcr.x - r_src.x).powi(2)
                 + (r_rcr.z + r_src.z).powi(2)).sqrt();
    let pulse = nuttall_pulse(fc, fs);

    let spec = ka_sum::Specs{
        r_src: r_src,
        r_rcr: r_rcr,
        fs: fs,
        c: c,
        duration: duration,
        tau_0: (d_img / c) - tau_buf,
        pulse: pulse,
    };

    let d_lim = (spec.tau_0 + spec.duration) * c;

    let (x0, x1) = bound_axes_x(&spec.r_src, &spec.r_rcr, d_lim, 0.5);
    let dy = bound_axes_y(&spec.r_src, &spec.r_rcr, d_lim, 0.5);

    let dr = (spec.r_src.x - spec.r_rcr.x).abs();

    let z_src2 = (spec.r_src.z - 0.5).powi(2);
    let z_rcr2 = (spec.r_rcr.z - 0.5).powi(2);

    let x_root = |x: F| {(x.powi(2) + z_src2).sqrt()
                       + ((x - dr).powi(2) + z_rcr2).sqrt()
                       - d_lim};
    println!("{}", x0);
    println!("{}", x1);
    println!("{}", dy);


    let x_a = M::range(x0, x1 + dx, dx);

    let eta = Array::<F, _>::zeros((x_a.raw_dim()[0], (2 as usize)));

    let surface = ka_sum::Surface1d{
        name: "Flat",
        x_a: x_a,
        eta: eta,
    };


    let ka_1d = ka_sum::ka_sum_1d(&spec, &surface);

}
