use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;
use array_lib::num_complex::Complex32;
use clap::Parser;
use unwrap_laplacian::bspline3d::BSpline3D;

#[derive(Debug,Parser)]
struct Args {
    bspline_json:PathBuf,
    out:PathBuf,
    #[arg(short)]
    x:usize,
    #[arg(short)]
    y:usize,
    #[arg(short)]
    z:usize,
}

fn main() {

    let args = Args::parse();

    let mut f = File::open(args.bspline_json.with_extension("json")).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    let bspline:BSpline3D = serde_json::from_str(&s).unwrap();

    let shape = [args.x, args.y, args.z];
    println!("rendering map with shape: {:?} ...",shape);
    let fit = bspline.render(&shape);
    let fit:Vec<_> = fit.into_iter().map(|x| Complex32::new(x,0.0 )).collect();
    write_cfl(args.out,&fit,ArrayDim::from_shape(&shape));

}