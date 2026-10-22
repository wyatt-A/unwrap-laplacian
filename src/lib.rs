pub mod region_growing;

use array_lib::ArrayDim;
use array_lib::num_complex::Complex32;
use dft_lib::common::{FftDirection, NormalizationType};
use rayon::prelude::*;

#[cfg(feature = "cuda")]
use dft_lib::cu_fft::cu_fftn as fftn;

#[cfg(feature = "fftw")]
use dft_lib::fftw_fft::fftw_fftn as fftn;

#[cfg(not(any(feature = "fftw", feature = "cuda")))]
use dft_lib::rs_fft::rs_fftn as fftn;

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;
    use super::*;
    use array_lib::io_nrrd::{write_nrrd, Encoding};

    #[test]
    fn test_unwrap_guass() {
        let dims:[usize;3] = [256,256,256];
        let vox_size = [1.,1.,1.];
        let wrapped = wrapped_gaussian(&dims,4.);
        let unwrapped = unwrap(&wrapped,&dims,&vox_size);
        write_nrrd("test_unwrapped",&unwrapped,ArrayDim::from_shape(&dims),None,true,Encoding::raw);
    }

    /// construct a phase wrapped image of a gaussian function
    fn wrapped_gaussian(dims:&[usize],pi_multiplier:f32) -> Vec<f32> {
        let var = ((dims[0].pow(2) + dims[1].pow(2) + dims[2].pow(2)) as f32).sqrt();
        let range = pi_multiplier * PI;
        let a_dims = ArrayDim::from_shape(&dims);
        let mut g = vec![0f32;a_dims.numel()];
        g.iter_mut().enumerate().for_each(|(i, g)|{
            let [x,y,z,..] = a_dims.calc_idx(i);
            let x = x as f32 - (dims[0]/2) as f32;
            let y = y as f32 - (dims[1]/2) as f32;
            let z = z as f32 - (dims[2]/2) as f32;
            let r_sq = (x*x + y*y + z*z);
            let x = (- r_sq / var).exp() * range;
            *g = (Complex32::i() * x).exp().to_polar().1
        });
        g
    }

}

pub fn unwrap(freq:&[f32], dims:&[usize], vox_size:&[f32]) -> Vec<f32> {
    // construct laplacian kernel in frequency domain
    let mut kern = laplacian_kernel(dims,vox_size);
    // avoid small imaginary component error
    kern.iter_mut().for_each(|x| x.im = 0.);

    let inv_kern:Vec<_> = kern.par_iter().map(|&x| 1./ x).map(|x|{
        if !x.is_finite() || x.norm() < 1e-10 {
            Complex32::ZERO
        } else {
            x
        }
    }).collect();

    // convert freq to complex values
    let mut freq:Vec<_> = freq.iter().map(|&x| Complex32::new(x,0.)).collect();

    let sin_f:Vec<_> = freq.par_iter().map(|&x| x.sin()).collect();
    let cos_f:Vec<_> = freq.par_iter().map(|&x| x.cos()).collect();

    let mut first_term = sin_f.clone();
    let mut second_term = cos_f.clone();

    fftn(&mut first_term, dims, FftDirection::Forward, NormalizationType::Inverse);
    fftn(&mut second_term, dims, FftDirection::Forward, NormalizationType::Inverse);

    first_term.iter_mut().zip(kern.iter()).for_each(|(x,&k)| *x = *x * k);
    second_term.iter_mut().zip(kern.iter()).for_each(|(x,&k)| *x = *x * k);

    fftn(&mut first_term, dims, FftDirection::Inverse, NormalizationType::Inverse);
    fftn(&mut second_term, dims, FftDirection::Inverse, NormalizationType::Inverse);

    first_term.iter_mut().zip(cos_f.iter()).for_each(|(x,&k)| *x = *x * k);
    second_term.iter_mut().zip(sin_f.iter()).for_each(|(x,&k)| *x = *x * k);

    first_term.iter_mut().zip(second_term.iter()).for_each(|(x,&y)| *x = *x - y);

    fftn(&mut first_term, dims, FftDirection::Forward, NormalizationType::Inverse);

    first_term.iter_mut().zip(inv_kern.into_iter()).for_each(|(x,k)| *x = *x * k);

    fftn(&mut first_term, dims, FftDirection::Inverse, NormalizationType::Inverse);

    first_term.into_par_iter().map(|x| x.re).collect()

}





/// constructs a laplacian kernel in the frequency domain
pub fn laplacian_kernel(dims:&[usize], vox_size:&[f32]) -> Vec<Complex32> {

    assert_eq!(dims.len(), vox_size.len());
    assert_eq!(dims.len(), 3);

    let a_dims = ArrayDim::from_shape(dims);

    let mut kern = vec![Complex32::ZERO; a_dims.numel()];

    let mut s_coords = [0,0,0];
    let center = [dims[0]/2, dims[1]/2, dims[2]/2];

    let coord = center;
    a_dims.ifft_shift_coords(&coord, &mut s_coords);
    let idx = a_dims.calc_addr(&s_coords);
    kern[idx].re = -2. * (1./vox_size[0].powi(2) + 1./vox_size[1].powi(2) + 1./vox_size[2].powi(2));

    let mut coord = center;
    coord[0] += 1;
    a_dims.ifft_shift_coords(&coord, &mut s_coords);
    let idx = a_dims.calc_addr(&s_coords);
    kern[idx].re = 1./ (vox_size[0] * vox_size[0]);

    let mut coord = center;
    coord[0] -= 1;
    a_dims.ifft_shift_coords(&coord, &mut s_coords);
    let idx = a_dims.calc_addr(&s_coords);
    kern[idx].re = 1./ (vox_size[0] * vox_size[0]);

    let mut coord = center;
    coord[1] += 1;
    a_dims.ifft_shift_coords(&coord, &mut s_coords);
    let idx = a_dims.calc_addr(&s_coords);
    kern[idx].re = 1./ (vox_size[1] * vox_size[1]);

    let mut coord = center;
    coord[1] -= 1;
    a_dims.ifft_shift_coords(&coord, &mut s_coords);
    let idx = a_dims.calc_addr(&s_coords);
    kern[idx].re = 1./ (vox_size[1] * vox_size[1]);

    let mut coord = center;
    coord[2] += 1;
    a_dims.ifft_shift_coords(&coord, &mut s_coords);
    let idx = a_dims.calc_addr(&s_coords);
    kern[idx].re = 1./ (vox_size[2] * vox_size[2]);

    let mut coord = center;
    coord[2] -= 1;
    a_dims.ifft_shift_coords(&coord, &mut s_coords);
    let idx = a_dims.calc_addr(&s_coords);
    kern[idx].re = 1./ (vox_size[2] * vox_size[2]);

    fftn(&mut kern,&dims,FftDirection::Forward,NormalizationType::None);

    kern
}