use kasea::*;
use criterion::{
    black_box, criterion_group, criterion_main, Criterion,
};
use ndarray::prelude::*;

fn criterion_benchmark(c: &mut Criterion) {

    let dx = 0.5;
    let x0 = -3.0;
    let x1 = 10.0;
    let y0 = -10.0;
    let y1 = 10.0;

    let x = Array::range(x0, x1 + dx, dx);
    let y = Array::range(y0, y1 + dx, dx);
    let z = Array::from_elem((x.raw_dim()[0], y.raw_dim()[0]), 0.0);
    let r0 = array![0.0, 0.0, -10.0];
    let max_r = 15.0;

    // create a benchmark group as we want to compare variants of a
    // particular function
    let mut group = c.benchmark_group("distance");
    group.bench_function("std", |b| {
        //b.iter(|| r_mag(black_box(&x), &y, &r0, &z));
        b.iter(|| xform(black_box(&x), &y, &r0, max_r, &z));
    });
    group.bench_function("ind", |b| {
        b.iter(|| xform_f(black_box(&x), &y, &r0, max_r, &z));
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
