use ndarray::prelude::*;
use kasea::{F, M, PI2};
use crate::ka_sum::{ka_sum_1d, Specs, Surface};

pub mod ka_sum;

fn main() {

    let dx: F = 0.05;
    let x0: F = -2.0;
    let x1: F = 12.0;
    let r_src = M::from_vec(vec![0.0, 0.0, -5.0]);
    let r_rcr = M::from_vec(vec![10.0, 0.0, -10.0]);

    let x_a = M::range(x0, x1 + dx, dx);

    let k_a = PI2 * M::from_vec(vec![2.0, 1.0, 0.5]);
    let r_max: F = 20.0;

    let eta = Surface::M::from_elem(x_a.raw_dim()[0], 0.0);
    let eta_x = Surface::M::from_elem(x_a.raw_dim()[0], 0.0);

    let spec = Specs{
        r_src: r_src,
        r_rcr: r_rcr,
        r_max: r_max,
        x_a: x_a,
        y_a: None,
        k_a: k_a,
        eta: eta,
        eta_x: eta_x,
        eta_y: None,
    };

    let sums2 = ka_sum_1d(spec);

    println!("{}", sums2)

}
