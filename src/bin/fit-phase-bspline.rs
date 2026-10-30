use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::num_complex::Complex32;
use clap::{arg, Parser};
use unwrap_laplacian::bspline3d::BSpline3D;
use unwrap_laplacian::polynomial_model::Polynomial3D;
use unwrap_laplacian::unwrap_rg::unwrap_rg;

#[derive(Debug,Parser)]
struct Args {
    /// input real-valued phase cfl
    cfl_in:PathBuf,
    mask_in:PathBuf,
    output_phase:PathBuf,

    /// magnitude weighting exponent. Default is 2.0 (magn^2) weighting
    #[clap(short,long,default_value_t=2.0)]
    gamma:f64,

    /// regularization for b-spline fit
    #[clap(short='r',long,default_value_t=0.01)]
    lambda:f64,

    /// controls control point spacing. Shrink factor of 2 means a control point every 2 voxels
    #[clap(short='s',long,default_value_t=1)]
    shrink_factor:usize,

    /// max iterations for b-spline fitting
    #[clap(short='i',long,default_value_t=500)]
    max_iter:usize,

    /// radius for seed initialization relative to central image sample
    #[clap(short,long,default_value_t=20)]
    seed_radius:usize,
}

fn main() {

    let args = Args::parse();

    println!("loading data...");
    let (x,x_dims) = read_cfl(&args.cfl_in);
    let (m,m_dims) = read_cfl(&args.mask_in);
    assert_eq!(x_dims.numel(),m_dims.numel());
    let shape = x_dims.shape3();

    assert_eq!(shape.iter().product::<usize>(),x.len(),"expected a single 3-D volume");

    let m:Vec<f32> = m.into_iter().map(|x| x.re).collect();
    let mut magn:Vec<f32> = x.iter().map(|x|x.norm()).collect();

    let mut unwrapped = x_dims.alloc(0.);

    println!("unwrapping phase ...");
    unwrap_rg(&x,&m,&mut unwrapped,&shape,args.seed_radius);

    magn.iter_mut().zip(unwrapped.iter_mut()).for_each(|(m,u)|{
        if u.is_nan() {
            *u = 0.0;
            *m = 0.0
        }
    });

    println!("fitting b-spline ...");
    let bspline = BSpline3D::fit(&unwrapped,&magn,&shape,[args.shrink_factor;3],2.0,args.lambda,args.max_iter).unwrap();
    let s = serde_json::to_string(&bspline).unwrap();
    let mut f = File::create(args.output_phase.with_extension("json")).unwrap();
    f.write_all(s.as_bytes()).unwrap();


}