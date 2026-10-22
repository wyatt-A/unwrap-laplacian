use std::path::PathBuf;
use array_lib::io_cfl::read_cfl;
use clap::Parser;
use mask::mask;


#[derive(Parser,Debug)]
struct Args {
    /// input .nhdr or .nrrd
    cfl_in:PathBuf,
    /// output .nhdr
    cfl_out:PathBuf,

    #[clap(short,long,default_value_t=20)]
    seed_radius:usize,

    #[clap(short,long)]
    debug:bool
}

fn main() {

    let args = Args::parse();

    let (x,dims) = read_cfl(args.cfl_in);

    let magn:Vec<_> = x.iter().map(|x|x.norm()).collect();
    let phase:Vec<_> = x.iter().map(|x|x.to_polar().1).collect();

    let mut m = vec![0.;dims.numel()];
    println!("solving for mask ...");
    mask(&magn,&mut m,dims.shape_ns());

    let mut unwrapped = vec![0.;dims.numel()];

    println!("unwrapping ...");
    unwrap_phase_3d(&phase,&magn,&m,&mut unwrapped,dims.shape_ns(),args.seed_radius);

    // find mean phase value and integer offset to normalize

    let a = unwrapped.iter().filter(|x| x.is_finite()).map(|x| *x as f64).sum::<f64>();
    let b = m.iter().map(|x| *x as f64).sum::<f64>();

    let p_mu = a / b;
    let n = (p_mu as f32 / (2.0 * PI)).round();
    unwrapped.iter_mut().for_each(|x| *x -= n*2.0*PI);

    write_nifti(args.cfl_out,&unwrapped,dims);
}



use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f32::consts::PI;
use array_lib::ArrayDim;
use array_lib::io_nifti::write_nifti;

#[derive(Clone, Copy, Debug)]
struct Candidate {
    priority: f32,
    voxel: usize,
    parent: usize,
}

// Reverse ordering: highest priority is popped first.
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
            && self.voxel == other.voxel
            && self.parent == other.parent
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| self.voxel.cmp(&other.voxel))
            .then_with(|| self.parent.cmp(&other.parent))
    }
}

#[inline]
fn wrap_phase(x: f32) -> f32 {
    (x + PI).rem_euclid(2.0 * PI) - PI
}

// Call f for each 6-connected neighbor.
#[inline]
fn neighbors(
    addr: usize,
    shape: [usize; 3],
    mut f: impl FnMut(usize),
) {
    let [nx, ny, nz] = shape;
    let plane = nx * ny;

    let x = addr % nx;
    let y = (addr / nx) % ny;
    let z = addr / plane;

    if x > 0      { f(addr - 1); }
    if x + 1 < nx { f(addr + 1); }

    if y > 0      { f(addr - nx); }
    if y + 1 < ny { f(addr + nx); }

    if z > 0      { f(addr - plane); }
    if z + 1 < nz { f(addr + plane); }
}

/// Quality-guided 3D phase unwrapping.
///
/// phase: wrapped phase in radians
/// quality: reliability score, higher is better
/// mask: nonzero = valid voxel
/// shape: [nx, ny, nz]
///
/// Returns unwrapped phase. Invalid voxels are NaN.
pub fn unwrap_phase_3d(
    phase: &[f32],
    quality: &[f32],
    mask: &[f32],
    dst:&mut [f32],
    shape: &[usize],
    seed_radius:usize,
) {

    let shape = [shape[0], shape[1], shape[2]];
    let dims = ArrayDim::from_shape(&shape);

    let n = dims.numel();

    assert!(shape.iter().all(|&d| d > 0));
    assert_eq!(phase.len(), n);
    assert_eq!(quality.len(), n);
    assert_eq!(mask.len(), n);

    let valid = |i: usize| {
        mask[i] > 0.
            && phase[i].is_finite()
            && quality[i].is_finite()
    };

    dst.fill(f32::NAN);
    let mut visited = vec![false; n];
    let mut heap = BinaryHeap::<Candidate>::new();

    let r = seed_radius as isize;
    let cx = shape[0]/2;
    let cy = shape[1]/2;
    let cz = shape[2]/2;

    let mut seed_region = vec![];
    for ix in cx as isize - r ..= cx as isize + r {
        for iy in cy as isize - r ..= cy as isize + r {
            for iz in cz as isize - r..=cz as isize + r {
                let addr = dims.calc_addr_signed(&[ix,iy,iz]);
                seed_region.push(addr);
            }
        }
    }

    // Select the most reliable seed.
    let seed = *seed_region.iter()
        .filter(|&i| valid(*i))
        .max_by(|&a, &b| quality[*a].total_cmp(&quality[*b])).unwrap();

    dst[seed] = phase[seed];
    visited[seed] = true;

    let push_neighbors = |
        parent: usize,
        heap: &mut BinaryHeap<Candidate>,
        visited: &[bool],
    | {
        neighbors(parent, shape, |voxel| {
            if valid(voxel) && !visited[voxel] {
                heap.push(Candidate {
                    priority: quality[parent].min(quality[voxel]),
                    voxel,
                    parent,
                });
            }
        });
    };

    push_neighbors(seed, &mut heap, &visited);

    while let Some(candidate) = heap.pop() {
        let j = candidate.voxel;
        let i = candidate.parent;

        // The same voxel can be inserted multiple times.
        if visited[j] {
            continue;
        }

        let delta = wrap_phase(phase[j] - phase[i]);

        dst[j] = dst[i] + delta;
        visited[j] = true;

        push_neighbors(j, &mut heap, &visited);
    }

}
