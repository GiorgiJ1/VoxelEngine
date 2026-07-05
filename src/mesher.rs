use crate::chunk::{Chunk, CHUNK_SIZE};
use crate::mesh::MeshData;
use crate::voxel::Voxel;

const SIZE: i32 = CHUNK_SIZE as i32;

fn sample(chunk: &Chunk, x: i32, y: i32, z: i32) -> Voxel {
    if x < 0 || y < 0 || z < 0 || x >= SIZE || y >= SIZE || z >= SIZE {
        Voxel::EMPTY
    } else {
        chunk.get(x as usize, y as usize, z as usize)
    }
}

pub fn greedy_mesh(chunk: &Chunk) -> MeshData {
    let mut mesh = MeshData::default();

    if chunk.is_empty() {
        return mesh;
    }

    for axis in 0..3 {
        let u = (axis + 1) % 3;
        let v = (axis + 2) % 3;

        let mut x = [0i32; 3];
        let mut q = [0i32; 3];
        q[axis] = 1;

        let mut mask: Vec<Option<(Voxel, bool)>> = vec![None; CHUNK_SIZE * CHUNK_SIZE];

        x[axis] = -1;
        while x[axis] < SIZE {
            let mut n = 0;
            for xv in 0..SIZE {
                for xu in 0..SIZE {
                    x[u] = xu;
                    x[v] = xv;

                    let a = sample(chunk, x[0], x[1], x[2]);
                    let b = sample(chunk, x[0] + q[0], x[1] + q[1], x[2] + q[2]);

                    mask[n] = match (a.is_empty(), b.is_empty()) {
                        (false, true) => Some((a, false)),
                        (true, false) => Some((b, true)),
                        _ => None,
                    };

                    n += 1;
                }
            }

            x[axis] += 1;

            let mut n = 0;
            for j in 0..CHUNK_SIZE {
                let mut i = 0;
                while i < CHUNK_SIZE {
                    if let Some(entry) = mask[n] {
                        let mut w = 1;
                        while i + w < CHUNK_SIZE && mask[n + w] == Some(entry) {
                            w += 1;
                        }

                        let mut h = 1;
                        'grow_height: while j + h < CHUNK_SIZE {
                            for k in 0..w {
                                if mask[n + k + h * CHUNK_SIZE] != Some(entry) {
                                    break 'grow_height;
                                }
                            }
                            h += 1;
                        }

                        let (voxel, backface) = entry;

                        x[u] = i as i32;
                        x[v] = j as i32;
                        let mut du = [0i32; 3];
                        let mut dv = [0i32; 3];
                        du[u] = w as i32;
                        dv[v] = h as i32;

                        emit_quad(&mut mesh, x, du, dv, axis, backface, voxel);

                        for l in 0..h {
                            for k in 0..w {
                                mask[n + k + l * CHUNK_SIZE] = None;
                            }
                        }

                        i += w;
                        n += w;
                    } else {
                        i += 1;
                        n += 1;
                    }
                }
            }
        }
    }

    mesh
}

fn emit_quad(
    mesh: &mut MeshData,
    x: [i32; 3],
    du: [i32; 3],
    dv: [i32; 3],
    axis: usize,
    backface: bool,
    voxel: Voxel,
) {
    let p0 = [x[0] as f32, x[1] as f32, x[2] as f32];
    let p1 = [(x[0] + du[0]) as f32, (x[1] + du[1]) as f32, (x[2] + du[2]) as f32];
    let p2 = [
        (x[0] + du[0] + dv[0]) as f32,
        (x[1] + du[1] + dv[1]) as f32,
        (x[2] + du[2] + dv[2]) as f32,
    ];
    let p3 = [(x[0] + dv[0]) as f32, (x[1] + dv[1]) as f32, (x[2] + dv[2]) as f32];

    let mut normal = [0.0f32; 3];
    normal[axis] = if backface { -1.0 } else { 1.0 };

    let verts = if backface {
        [p0, p3, p2, p1]
    } else {
        [p0, p1, p2, p3]
    };

    let base = mesh.positions.len() as u32;
    for p in verts {
        mesh.positions.push(p);
        mesh.normals.push(normal);
        mesh.voxel_ids.push(voxel.0);
    }

    mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}