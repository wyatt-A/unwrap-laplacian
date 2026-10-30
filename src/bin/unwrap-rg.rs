use std::path::PathBuf;
use array_lib::io_cfl::{read_cfl, write_cfl};
use clap::Parser;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f32::consts::PI;
use std::time::Instant;
use array_lib::ArrayDim;
use array_lib::cfl::num_traits::Zero;
use array_lib::num_complex::Complex32;
use unwrap_laplacian::unwrap_rg::unwrap_rg;

#[derive(Parser,Debug)]
struct Args {
    /// input complex image
    cfl_in:PathBuf,

    /// input mask for region growing algorithm
    cfl_mask:PathBuf,

    /// output real phase
    cfl_out:PathBuf,

    /// radius for seed initialization relative to central image sample
    #[clap(short,long,default_value_t=20)]
    seed_radius:usize,

    #[clap(short,long)]
    debug:bool
}

fn main() {
    let args = Args::parse();

    let (x,x_dims) = read_cfl(&args.cfl_in);
    let (m,m_dims) = read_cfl(&args.cfl_mask);
    assert_eq!(x_dims.numel(),m_dims.numel());
    let shape = x_dims.shape3();
    assert_eq!(shape.iter().product::<usize>(),x.len(),"expected a single 3-D volume");
    let m:Vec<_> = m.into_iter().map(|x|x.re).collect();

    let mut unwrapped = x_dims.alloc(0.);

    unwrap_rg(&x,&m,&mut unwrapped,&shape,args.seed_radius);

    let unwrapped:Vec<_> = unwrapped.into_iter().map(|x|Complex32::new(x,0.)).collect();
    write_cfl(args.cfl_out,&unwrapped,x_dims);

}
