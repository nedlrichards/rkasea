use eqsolver::{single_variable::FDNewton, finite_differences::central};
use crate::F;
use crate::geometry::Position;

pub fn bound_axes_x(r_src: &Position, r_rcr: &Position, d_lim: F, offset: F) -> (F, F) {
    // TODO: ignore dy for the moment
    let eps = 1e-5;
    let (dr, z_src2, z_rcr2) = relative_coord(r_src, r_rcr, offset);
    let x_root = |x: f64| {(x.powi(2) + z_src2).sqrt()
                       + ((x - dr).powi(2) + z_rcr2).sqrt()
                       - (d_lim as f64)};

    let x_start = FDNewton::new(x_root).with_tol(eps).solve(0.0).unwrap();
    let x_end = FDNewton::new(x_root).with_tol(eps).solve(dr).unwrap_or(0.0);
    (x_start as F, x_end as F)
}

pub fn bound_axes_y(r_src: &Position, r_rcr: &Position, d_lim: F, offset: F) -> F {
    // find y bounds
    let eps: f64 = 1e-5;
    let (dr, z_src2, z_rcr2) = relative_coord(r_src, r_rcr, offset);

    // position where y crosses d_lim
    let y_root = |x: f64, y: f64| {(x.powi(2) + y.powi(2) + z_src2).sqrt()
                         + ((x - dr).powi(2) + y.powi(2) + z_rcr2).sqrt()
                         - (d_lim as f64)};

    let y_lim = |x: f64| FDNewton::new(|y| y_root(x, y)).with_tol(eps).solve(1.0).unwrap();

    let d_y_lim = |x: f64| central(y_lim, x, eps);

    let test = FDNewton::new(d_y_lim).with_tol(1e-1).solve((dr / 2.0) as f64).unwrap();
    y_lim(test) as F
}

fn relative_coord(r_src: &Position, r_rcr: &Position, offset: F) -> (f64, f64, f64) {
    let dr = (r_src.x - r_rcr.x).abs() as f64;

    let z_src2 = (r_src.z + offset.copysign(r_src.z)).powi(2) as f64;
    let z_rcr2 = (r_rcr.z + offset.copysign(r_rcr.z)).powi(2) as f64;
    (dr, z_src2, z_rcr2)
}


