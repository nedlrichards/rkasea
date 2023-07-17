use ndarray::prelude::*;

use rkasea::ka_sum;
use rkasea::io_experiment::build_io;


fn main() {

    let io = build_io("/home/nedrichards/rust_proj/rkasea/experiments/flat.toml");

    let stat =  io.load_setup();
    let surface = io.load_surface(&stat);
    io.write_setup(&stat.pulse);

    let ka_1d = ka_sum::ka_sum_1d(&stat, &surface);

}
