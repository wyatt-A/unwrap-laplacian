use std::path::PathBuf;
use array_lib::io_cfl::read_cfl;
use clap::Parser;

#[derive(Parser,Debug)]
struct Args {
    /// input .nhdr or .nrrd
    cfl_in:PathBuf,
    /// output .nhdr
    cfl_out:PathBuf,

    #[clap(short,long)]
    debug:bool
}

fn main() {

    let args = Args::parse();

    let (x,dims) = read_cfl(args.cfl_in);





}