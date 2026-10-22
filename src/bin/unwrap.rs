use std::f32::consts::PI;
use std::path::PathBuf;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::num_complex::Complex32;
use clap::Parser;
use rayon::prelude::*;
use unwrap_laplacian::unwrap;

#[derive(Parser,Debug)]
struct Args {
    /// input .nhdr or .nrrd
    cfl_in:PathBuf,
    /// output .nhdr
    cfl_out:PathBuf,

    #[clap(short,long,default_value_t=1.)]
    x_spacing:f32,
    #[clap(short,long,default_value_t=1.)]
    y_spacing:f32,
    #[clap(short,long,default_value_t=1.)]
    z_spacing:f32,

    /// reconstruct phase using only the laplacian method
    #[clap(short,long)]
    laplacian_only: bool,

    #[clap(short,long)]
    debug:bool
}

fn main() {
    let args = Args::parse();
    let (x,dims) = read_cfl(&args.cfl_in);
    if args.debug {
        println!("cfl shape: {:?}",dims.shape_squeeze());
    }
    let [nx,ny,nz,..] = *dims.shape();
    assert_eq!(nx*ny*nz,dims.numel(),"input must only contain spatial dimensions");
    let mut phase:Vec<_> = x.par_iter().map(|x|x.to_polar().1).collect();
    let spacing = [args.x_spacing, args.y_spacing, args.z_spacing];
    let unwrapped = unwrap(&phase,&[nx,ny,nz],&spacing);

    let result = if args.laplacian_only {
        unwrapped
    }else {
        // construct integer map
        let int:Vec<_> = phase.par_iter().zip(unwrapped.par_iter()).map(|(phase,unwrapped)| {
            ((unwrapped - phase) / (2. * PI)).round()
        }).collect();

        // reconstruct the exact phase from integer map
        phase.par_iter_mut().zip(int.par_iter()).for_each(|(phase,int)| {
            *phase += 2. * PI * int;
        });
        phase
    };

    let out:Vec<_> = result.par_iter().map(|&x| Complex32::new(x,0.) ).collect();
    write_cfl(&args.cfl_out,&out,dims);
}