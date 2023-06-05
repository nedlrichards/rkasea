use num::complex::Complex;
#[cfg(feature = "bigger")] type CommonF = f64;
#[cfg(feature = "bigger")] use std::f64::consts::PI;
#[cfg(not(feature = "bigger"))] type CommonF = f32;
#[cfg(not(feature = "bigger"))] use std::f32::consts::PI;

#[inline(always)]
pub fn greens_3d(r: CommonF, k: CommonF) -> Complex<CommonF> {
    Complex::new(0.0, -r * k).exp() / (4.0 * PI * r)
}

#[inline(always)]
pub fn greens_3d_der(r: CommonF, k: CommonF) -> Complex<CommonF> {
    Complex::new(0.0, -k) * Complex::new(0.0, -r * k).exp() / (4.0 * PI * r)
}

#[inline(always)]
pub fn greens_2d(r: CommonF, k: CommonF) -> Complex<CommonF> {
    let z = r * k;
    (2.0 / (PI * z)).sqrt() * Complex::new(0.0, -z + PI / 4.0).exp()
}

#[inline(always)]
pub fn greens_2d_der(r: CommonF, k: CommonF) -> Complex<CommonF> {
    let z = r * k;
    Complex::new(0.0, -k * (2.0 / (PI * z)).sqrt()) * Complex::new(0.0, -z + 3.0 * PI / 4.0).exp()
}


