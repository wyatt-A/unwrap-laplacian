use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;
use array_lib::io_nrrd::{read_nrrd, write_nrrd, Encoding};
use clap::Parser;
use unwrap_laplacian::unwrap;

#[derive(Parser,Debug)]
struct Args {
    /// input .nhdr or .nrrd
    input:PathBuf,
    /// output .nhdr
    output:PathBuf,

    #[clap(long)]
    /// voxel size in mm
    /// Example: `--vox-size [0.03,0.03,0.03]`
    vox_size:Option<String>,
}

fn main() -> Result<(),String> {

    let args = Args::parse();

    let (x,a_dims,ref_header) = read_nrrd::<f32>(&args.input);

    // determine voxel spacing to use
    let vox_size = if let Some(vox_size) = &args.vox_size {
        let vox_size = parse_list_input::<f32>(vox_size,'[',']').expect("failed to parse vox_size");
        assert_eq!(vox_size.len(),3,"expected volume to be 3-D");
        vox_size
    }else if let Some(spacing) = &ref_header.space_directions {
        let vox_size = spacing.norms();
        assert_eq!(vox_size.len(),3,"expected volume to be 3-D");
        vox_size.into_iter().map(|x| x as f32).collect()
    }else {
        vec![1.,1.,1.]
    };

    let unwrapped = unwrap(&x,a_dims.shape_ns(),&vox_size);

    if let Some(ext) = args.output.extension() {
        match ext.to_str().unwrap() {
            "nhdr" => write_nrrd(&args.output,&unwrapped,a_dims,Some(&ref_header),false,Encoding::raw),
            "nrrd" => write_nrrd(&args.output,&unwrapped,a_dims,Some(&ref_header),true,Encoding::raw),
            _=> return Err(format!("unsupported extension {}",ext.to_str().unwrap()))
        }
    }else {
        write_nrrd(&args.output,&unwrapped,a_dims,Some(&ref_header),false,Encoding::raw);
    }
    Ok(())
}

pub fn parse_list_input<T>(s: &str, open: char, close: char) -> Result<Vec<T>, String>
where
    T: FromStr,
    <T as FromStr>::Err: Display,
{
    let start = s.find(open).ok_or_else(|| format!("missing open delimiter '{}'", open))?;
    let end_rel = s[start + 1..]
        .find(close)
        .ok_or_else(|| format!("missing close delimiter '{}'", close))?;
    let end = start + 1 + end_rel; // make it absolute

    let inner = &s[start + 1..end];

    let vals = inner
        .split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<T>().map_err(|e| format!("failed to parse '{}': {e}", t)))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(vals)
}