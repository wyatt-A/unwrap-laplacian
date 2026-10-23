use std::path::PathBuf;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::num_complex::Complex32;
use clap::Parser;
use rayon::prelude::*;

#[derive(Parser,Debug)]
struct Args {
    cfl_in: PathBuf,
    phase_out: PathBuf,
}

fn main() {

    let args = Args::parse();
    let (x, dims) = read_cfl(args.cfl_in);
    let out:Vec<_> = x.into_par_iter().map(|x|Complex32::new(x.to_polar().1,0.)).collect();
    write_cfl(args.phase_out, &out, dims);

}