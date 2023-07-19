use ndarray::prelude::*;
use crate::{F, M, PI};

#[derive(Clone)]
pub struct Pulse {
    pub fc: F,
    pub fs: F,
    pub signal: M,
}

impl Pulse {
    pub fn t_axis(&self) -> Array1<F> {
        Array::range(0.0, (self.signal.len() as F) / self.fs, 1.0 / self.fs)
    }
    pub fn f_axis(&self) -> Array1<F> {
        Array::range(0.0, (self.signal.len() / 2 + 1) as F, 1.0) * self.fs
    }

}

pub fn nuttall_pulse(fc: F) -> Pulse {

    let nuttall: M = array![0.4243801, -0.4973406, 0.0782793];
    let num_cycles: F = 5.0;

    let f_cut = 1.9 * fc;  // estimate of Q for pulse
    let fs = 2.0 * f_cut;

    let mut num_samples = (fs * num_cycles / fc).ceil();
    if (num_samples % 2.0) > 0.5 {num_samples += 1.0;}

    let samps: M = Array::range(0., num_samples, 1.);
    let xmitt = samps.map(|&s| (2.0 * PI * fc * s / fs).cos()
                          * nuttall.view().into_iter().enumerate()
                          .fold(0.0, |acc, (i, a)| acc + a * ((i as F) * 2.0 * PI * s / num_samples).cos()));

    Pulse{fc:fc, fs:fs, signal: xmitt}
}

