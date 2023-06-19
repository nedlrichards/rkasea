use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use plotters::prelude::*;
use std::iter::zip;
use rkasea::xmitt::nuttall_pulse;
use rkasea::{F, M};
use realfft::RealFftPlanner;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let fc: F = 3e3;
    let fs: F = 20e3;

    let xmitt = nuttall_pulse(fc);
    let dt = 1.0 / fs;
    let t_a = Array1::range(0.0, dt * (xmitt.signal.raw_dim()[0] as F), dt);

    let t_min: f32 = t_a[0];
    let t_max: f32 = t_a[t_a.len() - 1];

    let iter = zip(t_a.map(|t| t * 1e3).to_vec(), xmitt.signal.to_vec());

    let root = BitMapBackend::new("/home/nedrichards/rust_proj/rkasea/figures/pulse.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(512);

    let mut chart = ChartBuilder::on(&upper)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(t_min*1e3..t_max*1e3, -1.5f32..1.5f32)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(iter, &RED))?
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED))
        .label("xmitt");

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    let num_f = (t_a.len() as F) / 2.0 + 1.0;
    let fs = ((t_a.len() as F) - 1.0) / (t_a[t_a.len() - 1] - t_a[0]);
    let f_scale = fs / (t_a.len() as F);

    let f_a: M = Array::range(0., num_f * f_scale, f_scale);
    let mut planner = RealFftPlanner::<F>::new();
    let fft = planner.plan_fft_forward(t_a.len());
    let mut data = xmitt.signal.to_vec();
    let mut output = fft.make_output_vec();
    fft.process(&mut data, &mut output).unwrap();

    let f_db: M = output.into_iter().map(|a| (20.0 * (a.norm() + 1e-8).log10())).collect();
    let f_max = f_db.max_skipnan();
    let f_db = f_db.map(|f| f - f_max);

    let iter = zip(f_a.map(|f| f / 1e3).to_vec(), f_db);

    let mut chart = ChartBuilder::on(&lower)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..(f_a[f_a.len() - 1] / 1e3 as f32), -100f32..5f32)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(iter, &RED))?
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED))
        .label("xmitt");

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;



    root.present()?;

    Ok(())
}
