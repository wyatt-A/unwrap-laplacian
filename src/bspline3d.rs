use std::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BSpline3D {
    pub shape: [usize; 3],
    pub spacing: [usize; 3],
    pub grid_shape: [usize; 3],
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub enum GridMapping {
    AlignEndpoints,
    AlignVoxelCenters,
}

#[derive(Debug, Clone)]
pub struct FitError(pub String);
impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.0) }
}
impl std::error::Error for FitError {}

fn count(shape: [usize; 3]) -> Result<usize, FitError> {
    shape.into_iter().try_fold(1usize, |a, b| a.checked_mul(b).ok_or_else(|| FitError("size overflow".into())))
}
fn basis(t: f64) -> [f64; 4] {
    let t2 = t*t;
    let t3 = t2*t;
    [ (1.0-t).powi(3)/6.0,
        (3.0*t3-6.0*t2+4.0)/6.0,
        (-3.0*t3+3.0*t2+3.0*t+1.0)/6.0,
        t3/6.0 ]
}
fn axis_weights(i: usize, spacing: usize) -> ([usize;4], [f64;4]) {
    let q = i / spacing;
    let t = (i % spacing) as f64 / spacing as f64;
    ([q,q+1,q+2,q+3], basis(t))
}
fn stencil(x:usize,y:usize,z:usize, spacing:[usize;3], grid:[usize;3]) -> [(usize,f64);64] {
    let (ix,wx)=axis_weights(x,spacing[0]);
    let (iy,wy)=axis_weights(y,spacing[1]);
    let (iz,wz)=axis_weights(z,spacing[2]);
    let mut out=[(0usize,0.0f64);64];
    let mut n=0;
    for k in 0..4 { for j in 0..4 { for i in 0..4 {
        let idx=ix[i]+grid[0]*(iy[j]+grid[1]*iz[k]);
        out[n]=(idx,wx[i]*wy[j]*wz[k]);
        n+=1;
    }}}
    out
}
fn dot(a:&[f64],b:&[f64])->f64 {a.iter().zip(b).map(|(x,y)|x*y).sum()}

impl BSpline3D {
    pub fn fit(
        phase:&[f32], magnitude:&[f32], shape:&[usize], spacing:[usize;3],
        gamma:f64, lambda:f64, max_iter:usize,
    )->Result<Self,FitError> {
        let shape = [shape[0], shape[1], shape[2]];
        let n=count(shape)?;
        if n==0 || phase.len()!=n || magnitude.len()!=n {return Err(FitError("invalid image shape or input lengths".into()));}
        if spacing.iter().any(|&s|s==0) {return Err(FitError("spacing must be positive".into()));}
        if !gamma.is_finite() || gamma<0.0 || !lambda.is_finite() || lambda<0.0 {return Err(FitError("gamma and lambda must be finite and nonnegative".into()));}
        if max_iter==0 {return Err(FitError("max_iter must be positive".into()));}
        let mut grid=[0;3];
        for d in 0..3 {grid[d]=(shape[d]-1)/spacing[d]+4;}
        let nc=count(grid)?;
        let max_mag=magnitude.iter().copied().filter(|v|v.is_finite() && *v>0.0).fold(0.0f32,f32::max) as f64;
        if max_mag==0.0 {return Err(FitError("no positive finite magnitudes".into()));}
        let mut weights=vec![0.0f64;n];
        let mut valid=0usize;
        for v in 0..n {
            if phase[v].is_finite() && magnitude[v].is_finite() && magnitude[v]>0.0 {
                weights[v]=(magnitude[v] as f64/max_mag).powf(gamma);
                valid+=1;
            }
        }
        if valid==0 {return Err(FitError("no valid phase/magnitude pairs".into()));}
        let mut model=Self {shape,spacing,grid_shape:grid,coefficients:vec![0.0;nc]};
        let mut rhs=vec![0.0;nc];
        let mut diagonal=vec![0.0;nc];
        let [nx,ny,nz]=shape;
        for z in 0..nz {for y in 0..ny {for x in 0..nx {
            let v=x+nx*(y+ny*z);
            let w=weights[v];
            if w==0.0 {continue;}
            for (id,b) in stencil(x,y,z,spacing,grid) {
                rhs[id]+=w*b*phase[v] as f64;
                diagonal[id]+=w*b*b;
            }
        }}}
        // Add diagonal of lambda * L^T L, where L is the edge-difference operator.
        let [gx,gy,gz]=grid;
        for z in 0..gz {for y in 0..gy {for x in 0..gx {
            let i=x+gx*(y+gy*z);
            if x+1<gx {diagonal[i]+=lambda;diagonal[i+1]+=lambda;}
            if y+1<gy {diagonal[i]+=lambda;diagonal[i+gx]+=lambda;}
            if z+1<gz {diagonal[i]+=lambda;diagonal[i+gx*gy]+=lambda;}
        }}}
        let apply=|src:&[f64],dst:&mut[f64]| {
            dst.fill(0.0);
            for z in 0..nz {for y in 0..ny {for x in 0..nx {
                let v=x+nx*(y+ny*z);
                let w=weights[v];
                if w==0.0 {continue;}
                let st=stencil(x,y,z,spacing,grid);
                let value: f64=st.iter().map(|&(i,b)|src[i]*b).sum();
                for (i,b) in st {dst[i]+=w*b*value;}
            }}}
            if lambda>0.0 {
                for z in 0..gz {for y in 0..gy {for x in 0..gx {
                    let i=x+gx*(y+gy*z);
                    if x+1<gx {let j=i+1;let t=lambda*(src[i]-src[j]);dst[i]+=t;dst[j]-=t;}
                    if y+1<gy {let j=i+gx;let t=lambda*(src[i]-src[j]);dst[i]+=t;dst[j]-=t;}
                    if z+1<gz {let j=i+gx*gy;let t=lambda*(src[i]-src[j]);dst[i]+=t;dst[j]-=t;}
                }}}
            }
        };
        let mut r=rhs.clone();
        let mut zz=vec![0.0;nc];
        for i in 0..nc {zz[i]=r[i]/diagonal[i].max(1e-20);}
        let mut p=zz.clone();
        let mut rz=dot(&r,&zz);
        let rhs_norm=dot(&rhs,&rhs).sqrt();
        let tol=1e-6*rhs_norm.max(1e-30);
        let mut ap=vec![0.0;nc];
        let mut converged=dot(&r,&r).sqrt()<=tol;
        for it in 0..max_iter {
            println!("iter {}",it);
            if converged {break;}
            apply(&p,&mut ap);
            let denom=dot(&p,&ap);
            if !denom.is_finite() || denom<=0.0 {return Err(FitError("CG breakdown: non-positive curvature (try lambda > 0)".into()));}
            let alpha=rz/denom;
            for i in 0..nc {model.coefficients[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
            let resid = dot(&r, &r).sqrt();
            println!("r: {}, tol: {}", resid,tol);
            if resid <=tol {converged=true;break;}
            for i in 0..nc {zz[i]=r[i]/diagonal[i].max(1e-20);}
            let next=dot(&r,&zz);
            let beta=next/rz;
            for i in 0..nc {p[i]=zz[i]+beta*p[i];}
            rz=next;
        }
        if !converged {return Err(FitError("CG did not converge within max_iter; increase iterations or regularization".into()));}
        Ok(model)
    }
    pub fn evaluate(&self,x:usize,y:usize,z:usize)->f64 {
        assert!(x<self.shape[0] && y<self.shape[1] && z<self.shape[2]);
        stencil(x,y,z,self.spacing,self.grid_shape).iter().map(|&(i,b)|self.coefficients[i]*b).sum()
    }

    pub fn evaluate_at(
        &self,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        let coords = [x, y, z];

        let mut indices = [[0usize; 4]; 3];
        let mut weights = [[0.0f64; 4]; 3];

        for d in 0..3 {
            let coord = coords[d]
                .clamp(0.0, (self.shape[d] - 1) as f64);

            let u = coord / self.spacing[d] as f64;

            let base = u.floor() as isize;
            let t = u - base as f64;

            weights[d] = cubic_weights(t);

            for k in 0..4 {
                // One control point before the current cell.
                let i = base + k as isize - 1;

                indices[d][k] = i.clamp(
                    0,
                    self.grid_shape[d] as isize - 1,
                ) as usize;
            }
        }

        let [gx, gy, _gz] = self.grid_shape;

        let mut value = 0.0;

        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    let ix = indices[0][i];
                    let iy = indices[1][j];
                    let iz = indices[2][k];

                    let addr = ix + gx * (iy + gy * iz);

                    let w = weights[0][i]
                        * weights[1][j]
                        * weights[2][k];

                    value += w * self.coefficients[addr];
                }
            }
        }

        value
    }

    // pub fn render(&self,shape:&[usize])->Vec<f32> {
    //     let shape = [shape[0], shape[1], shape[2]];
    //     assert_eq!(shape,self.shape,"render shape must match fitted shape");
    //     let [nx,ny,nz]=shape;
    //     let mut out=vec![0.0;nx*ny*nz];
    //     for z in 0..nz {for y in 0..ny {for x in 0..nx {
    //         out[x+nx*(y+ny*z)]=self.evaluate(x,y,z) as f32;
    //     }}}
    //     out
    // }

    pub fn render_with_mapping(
        &self,
        dst_shape: [usize; 3],
        mapping: GridMapping,
    ) -> Vec<f32> {
        let [nx, ny, nz] = dst_shape;

        let mut output = vec![0.0f32; nx * ny * nz];

        for z in 0..nz {
            let zz = map_coordinate(
                z,
                self.shape[2],
                nz,
                mapping,
            );

            for y in 0..ny {
                let yy = map_coordinate(
                    y,
                    self.shape[1],
                    ny,
                    mapping,
                );

                for x in 0..nx {
                    let xx = map_coordinate(
                        x,
                        self.shape[0],
                        nx,
                        mapping,
                    );

                    let addr = x + nx * (y + ny * z);

                    output[addr] =
                        self.evaluate_at(xx, yy, zz) as f32;
                }
            }
        }

        output
    }

    pub fn render(&self, dst_shape: &[usize]) -> Vec<f32> {

        let dst_shape = [dst_shape[0], dst_shape[1], dst_shape[2]];

        self.render_with_mapping(
            dst_shape,
            GridMapping::AlignVoxelCenters,
        )
    }
}

fn map_coordinate(
    index: usize,
    src_size: usize,
    dst_size: usize,
    mapping: GridMapping,
) -> f64 {
    match mapping {
        GridMapping::AlignEndpoints => {
            if src_size <= 1 || dst_size <= 1 {
                0.0
            } else {
                index as f64
                    * (src_size - 1) as f64
                    / (dst_size - 1) as f64
            }
        }

        GridMapping::AlignVoxelCenters => {
            if src_size <= 1 {
                return 0.0;
            }

            let x = (index as f64 + 0.5)
                * src_size as f64
                / dst_size as f64
                - 0.5;

            x.clamp(0.0, (src_size - 1) as f64)
        }
    }
}

fn cubic_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;

    [
        (1.0 - 3.0*t + 3.0*t2 - t3) / 6.0,
        (4.0 - 6.0*t2 + 3.0*t3) / 6.0,
        (1.0 + 3.0*t + 3.0*t2 - 3.0*t3) / 6.0,
        t3 / 6.0,
    ]
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn constant_phase() {
//         let shape=[9,8,7];
//         let n=shape.iter().product();
//         let p=vec![1.75f32;n];
//         let m=vec![1.0f32;n];
//         let fit=BSpline3D::fit(&p,&m,shape,[4,4,4],2.0,0.01,400).unwrap();
//         assert!(fit.render(shape).iter().all(|v|(*v-1.75).abs()<1e-3));
//     }
// }
