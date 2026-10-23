use std::fs::read;
use std::path::PathBuf;
use array_lib::io_cfl::{read_cfl, write_cfl};
use clap::Parser;
use rayon::prelude::*;

#[derive(Parser,Debug)]
struct Args {
    cfl_a: PathBuf,
    cfl_b: PathBuf,
    out: PathBuf,
}

fn main() {

    let args = Args::parse();

    let (a,a_dims) = read_cfl(args.cfl_a);
    let (b,b_dims) = read_cfl(args.cfl_b);
    assert_eq!(a_dims.numel(), b_dims.numel());

    let mut out:Vec<_> = a.par_iter().zip(b.par_iter()).map(|(a,b)| {
        *b * a.conj()
    }).collect();

    out.par_iter_mut().for_each(|a| {
        *a = *a / a.norm();
    });

    write_cfl(args.out,&out,a_dims);

}