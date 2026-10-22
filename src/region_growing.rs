

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f32::consts::PI;

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
    mask: &[u8],
    shape: [usize; 3],
) -> Vec<f32> {
    let n = shape.iter().product::<usize>();

    assert!(shape.iter().all(|&d| d > 0));
    assert_eq!(phase.len(), n);
    assert_eq!(quality.len(), n);
    assert_eq!(mask.len(), n);

    let valid = |i: usize| {
        mask[i] != 0
            && phase[i].is_finite()
            && quality[i].is_finite()
    };

    let mut output = vec![f32::NAN; n];
    let mut visited = vec![false; n];
    let mut heap = BinaryHeap::<Candidate>::new();

    // Select the most reliable seed.
    let seed = (0..n)
        .filter(|&i| valid(i))
        .max_by(|&a, &b| quality[a].total_cmp(&quality[b]));

    let Some(seed) = seed else {
        return output;
    };

    output[seed] = phase[seed];
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

        output[j] = output[i] + delta;
        visited[j] = true;

        push_neighbors(j, &mut heap, &visited);
    }

    output
}
