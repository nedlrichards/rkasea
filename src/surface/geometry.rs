use ndarray::{Array, Array1, Array2};
use crate::{F};

pub struct Surface1d {
    pub name: String,
    pub dx: F,
    pub x0: F,
    pub num_x: i64,
}

impl Surface1d {
    pub fn x_axis(&self) -> Array1<F> {
        Array::range(self.x0, (self.num_x as F) * self.dx, self.dx)
    }
    pub fn eta(&self) -> Array2<F> {
        Array::zeros([self.num_x as usize, 2])
    }

}
