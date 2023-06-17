use ndarray::prelude::*;
use crate::{F, M, PI};

pub fn nuttall_pulse(fc: F, fs: F) -> M {

    let nuttall: M = array![0.4243801, -0.4973406, 0.0782793];
    let num_cycles: F = 5.;
    let mut num_samples = (fs * num_cycles / fc).ceil();
    if (num_samples % 2.0) > 0.5 {num_samples += 1.0;}

    let samps: M = Array::range(0., num_samples, 1.);
    let xmitt = samps.map(|&s| (2.0 * PI * fc * s / fs).cos()
                          * nuttall.view().into_iter().enumerate()
                          .fold(0.0, |acc, (i, a)| acc + a * ((i as F) * 2.0 * PI * s / num_samples).cos()));


    xmitt
}

