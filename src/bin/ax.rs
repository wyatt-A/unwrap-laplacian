use std::path::PathBuf;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::num_complex::Complex32;
use clap::Parser;
use rayon::prelude::*;

#[derive(Parser,Debug)]
struct Args {
    cfl_x: PathBuf,
    cfl_a: PathBuf,
    
    scale_a:f32,

    #[clap(short)]
    conj_a:bool,

    result: PathBuf,
}

fn main() {

    let args = Args::parse();
    let (x, ..) = read_cfl(args.cfl_x);
    let (a, dims) = read_cfl(args.cfl_a);

    let n = x.len();
    let mut result = vec![Complex32::ZERO; n];

    result.par_iter_mut().enumerate().for_each(|(i, r)| {

        let a = if args.conj_a {
            a[i].conj()
        }else {
            a[i]
        };

        *r = args.scale_a * a * x[i]
    });

    write_cfl(args.result, &result, dims);

}