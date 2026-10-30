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

pub fn unwrap_rg(x:&[Complex32], mask:&[f32], dst:&mut [f32], shape:&[usize], seed_radius:usize) {

    let [nx,ny,nz] = [shape[0],shape[1],shape[2]];

    let n_vox = nx*ny*nz;

    assert_eq!(x.len(), n_vox);
    assert_eq!(mask.len(), n_vox);
    assert_eq!(dst.len(), n_vox);

    dst.fill(0.0);

    let mut magn = vec![0.; n_vox];
    let mut phase = vec![0.; n_vox];

    magn.par_iter_mut().zip(phase.par_iter_mut()).enumerate().for_each(|(i,(magn,phase))|{
        (*magn,*phase) = x[i].to_polar();
    });

    unwrap_phase_3d(&phase,&magn,&mask,dst,&shape,seed_radius);

    let a = dst.par_iter().filter(|x| x.is_finite()).map(|x| *x as f64).sum::<f64>();
    let b = mask.par_iter().map(|x| *x as f64).sum::<f64>();
    let p_mu = a / b;
    let n = (p_mu as f32 / (2.0 * PI)).round();
    dst.iter_mut().for_each(|x| *x -= n*2.0*PI);

}


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
