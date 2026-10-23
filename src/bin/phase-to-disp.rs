use std::f32::consts::PI;
use std::path::PathBuf;
use array_lib::ArrayDim;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::io_nifti::write_nifti;
use array_lib::num_complex::{Complex32, ComplexFloat};
use clap::Parser;
use rayon::prelude::*;

#[derive(Parser,Debug)]
struct Args {
    /// input phase map (real channel of cfl)
    cfl_phase: PathBuf,
    /// output displacement map with unit spacing (real channel of cfl)
    output_disp: PathBuf,

    /// echo time of GRE in milliseconds
    echo_time_ms:f32,

    /// sectral width of readout in kHz
    spectral_width_khz:f32,

    /// dimension associated with readout dimension (i.e. 0 -> x, 1 -> y etc.)
    readout_dim: usize,

    /// displacement scale factor. For opposite polarity scans, this would be 0.5 and -0.5, respectively
    scale: f32,
}


fn main() {

    let args = Args::parse();

    let (p,dims) = read_cfl(args.cfl_phase);
    let p:Vec<f32> = p.into_par_iter().map(|p| p.re).collect();

    let shape = dims.shape3();
    assert_eq!(shape.iter().product::<usize>(),dims.numel(),"expected a single 3-D image");
    let nx = dims.get(args.readout_dim);
    let bw_px = args.spectral_width_khz * 1e3 / nx as f32;

    println!("echo time: {} ms",args.echo_time_ms);
    println!("n readout: {nx}");
    println!("bw pixel: {bw_px} Hz/px");

    let disp_x_px:Vec<_> = p.par_iter().map(|p|{
        if p.is_finite() {
            args.scale * *p / (2.0 * PI * args.echo_time_ms * 1e-3 * bw_px)
        }else {
            0.
        }
    } ).collect();

    let dx:Vec<_> = disp_x_px.par_iter().map(|&x| Complex32::new(x,0.)).collect();
    write_cfl(args.output_disp,&dx,dims);

}