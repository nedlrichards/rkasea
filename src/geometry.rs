use ndarray::prelude::*;

pub fn r_mag(x: &Array1<f64>, y: &Array1<f64>, r0: &Array1<f64>, z: &Array2<f64>) -> Array2<f64> {

    let x_sq = x.map(|v| f64::powi(*v - r0[0], 2));
    let y_sq = y.map(|v| f64::powi(*v - r0[1], 2));
    let mut res = z.map(|v| f64::powi(*v - r0[2], 2));

    azip!((z_row in res.rows_mut(), x2 in &x_sq) {
        azip!((z2 in z_row, y2 in &y_sq) {
            *z2 = (*z2 + x2 + y2).sqrt()
        })
    });

    res
}

pub fn proj(x: &Array1<f64>, y: &Array1<f64>, r0: &Array1<f64>, r: &Array2<f64>,
            z: &Array2<f64>, z_x: &Array2<f64>, z_y: &Array2<f64>,) -> Array2<f64> {

    let dx = x.map(|v| *v - r0[0]);
    let dy = y.map(|v| *v - r0[1]);
    let mut res = z.map(|v| *v - r0[2]);

    azip!((z_r in res.rows_mut(), z_x_r in z_x.rows(), z_y_r in z_y.rows(), r_r in r.rows(), x in &dx) {
        azip!((z_i in z_r, z_x_i in &z_x_r, z_y_i in &z_y_r, y in &dy, r_i in &r_r) {
            *z_i = ((-z_x_i * x) + (-z_y_i * y) + *z_i) / r_i
        })
    });

    res
}
pub fn xform(x: &Array1<f64>, y: &Array1<f64>, r0: &Array1<f64>, max_r: f64, eta: &Array2<f64>) -> Array2<f64> {

    let all_r = r_mag(&x, &y, &r0, &eta);

    let ncols = 2;
    let mut nrows = 0;
    // count number of in values
    for r in all_r.iter() { if *r < max_r { nrows += 1 }}

    let mut arr1 = Array2::<f64>::zeros((nrows, ncols));
    let mut i = 0;
    ndarray::Zip::from(&all_r).and(eta).for_each(|r, z| {
        if *r < max_r {
            arr1.slice_mut(s![i, ..]).assign(&ArrayView::from(&[*r, *z]));
            i += 1;
        }
    });
    arr1
}

pub fn xform_f(x: &Array1<f64>, y: &Array1<f64>, r0: &Array1<f64>, max_r: f64, eta: &Array2<f64>) -> Array2<f64> {

    let all_r = r_mag(&x, &y, &r0, &eta);

    let ncols = 2;
    let mut data = Vec::new();
    let mut nrows = 0;

    ndarray::Zip::from(&all_r).and(eta).for_each(|r, z| {
        if *r < max_r {
            data.extend_from_slice(&[*r, *z]);
            nrows += 1;
        }
    });

    let arr = Array2::from_shape_vec((nrows, ncols), data).unwrap();
    arr

}
