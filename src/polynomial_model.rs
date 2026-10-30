
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct Polynomial3D {
    pub degree: usize,
    pub powers: Vec<[usize; 3]>,
    pub coefficients: Vec<f64>,
}

impl Polynomial3D {
    pub fn fit(
        phase: &[f32],
        magnitude: &[f32],
        shape: &[usize],
        degree: usize,
        gamma: f64,
    ) -> Result<Self, String> {
        let [nx, ny, nz] = [shape[0],shape[1],shape[2]];
        let nvox = nx * ny * nz;

        if phase.len() != nvox || magnitude.len() != nvox {
            return Err("Input dimensions do not match".into());
        }
        if !gamma.is_finite() || gamma < 0.0 {
            return Err("Invalid magnitude exponent".into());
        }

        // Enumerate monomials of total degree <= degree.
        let mut powers = Vec::new();
        for total in 0..=degree {
            for i in 0..=total {
                for j in 0..=(total - i) {
                    let k = total - i - j;
                    powers.push([i, j, k]);
                }
            }
        }

        let nc = powers.len();
        let mut ata = DMatrix::<f64>::zeros(nc, nc);
        let mut atb = DVector::<f64>::zeros(nc);

        let max_mag = magnitude
            .iter()
            .filter(|m| m.is_finite() && **m > 0.0)
            .fold(0.0_f64, |a, &b| a.max(b as f64));

        if max_mag == 0.0 {
            return Err("No positive magnitude values".into());
        }

        // Coordinates normalized to [-1, 1].
        let norm = |i: usize, n: usize| -> f64 {
            if n <= 1 {
                0.0
            } else {
                2.0 * i as f64 / (n - 1) as f64 - 1.0
            }
        };

        let mut nvalid = 0usize;

        // x is the fastest-varying dimension.
        for z in 0..nz {
            let zz = norm(z, nz);
            for y in 0..ny {
                let yy = norm(y, ny);
                for x in 0..nx {
                    let xx = norm(x, nx);
                    let addr = x + nx * (y + ny * z);

                    let phi = phase[addr] as f64;
                    let mag = magnitude[addr] as f64;

                    if !phi.is_finite()
                        || !mag.is_finite()
                        || mag <= 0.0
                    {
                        continue;
                    }

                    let w = (mag / max_mag).powf(gamma);
                    if w <= 0.0 {
                        continue;
                    }

                    let basis: Vec<f64> = powers
                        .iter()
                        .map(|&[i, j, k]| {
                            xx.powi(i as i32)
                                * yy.powi(j as i32)
                                * zz.powi(k as i32)
                        })
                        .collect();

                    for i in 0..nc {
                        atb[i] += w * basis[i] * phi;
                        for j in 0..=i {
                            ata[(i, j)] +=
                                w * basis[i] * basis[j];
                        }
                    }

                    nvalid += 1;
                }
            }
        }

        if nvalid < nc {
            return Err("Insufficient valid voxels".into());
        }

        // Complete symmetric matrix.
        for i in 0..nc {
            for j in 0..i {
                ata[(j, i)] = ata[(i, j)];
            }
        }

        // Solve using Cholesky decomposition.
        let chol = ata.cholesky().ok_or(
            "Polynomial system is singular or ill-conditioned"
        )?;

        let coeff = chol.solve(&atb);

        Ok(Self {
            degree,
            powers,
            coefficients: coeff.iter().copied().collect(),
        })
    }

    pub fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        self.powers
            .iter()
            .zip(&self.coefficients)
            .map(|(&[i, j, k], &c)| {
                c * x.powi(i as i32)
                    * y.powi(j as i32)
                    * z.powi(k as i32)
            })
            .sum()
    }

    pub fn render(&self, shape: &[usize]) -> Vec<f32> {
        let [nx, ny, nz] = [shape[0],shape[1],shape[2]];
        let mut output = vec![0.0f32; nx * ny * nz];

        let norm = |i: usize, n: usize| -> f64 {
            if n <= 1 {
                0.0
            } else {
                2.0 * i as f64 / (n - 1) as f64 - 1.0
            }
        };

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let addr = x + nx * (y + ny * z);

                    output[addr] = self.evaluate(
                        norm(x, nx),
                        norm(y, ny),
                        norm(z, nz),
                    ) as f32;
                }
            }
        }

        output
    }
}
