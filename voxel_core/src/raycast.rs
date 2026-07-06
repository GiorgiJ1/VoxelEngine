use glam::{IVec3, Vec3};

use crate::chunk::{Chunk, CHUNK_SIZE};

/// Result of a ray hitting a solid voxel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RayHit {
    /// The solid voxel the ray hit (in chunk-local coordinates).
    pub voxel: IVec3,
    /// The empty cell just before `voxel` along the ray, i.e. where a new
    /// voxel should go if the user wants to build onto this face. `None`
    /// if the ray started already inside a solid voxel (edge case) or if
    /// that cell would fall outside the chunk.
    pub place_at: Option<IVec3>,
}

fn ray_aabb(origin: Vec3, dir: Vec3, box_min: Vec3, box_max: Vec3) -> Option<(f32, f32)> {
    let o = origin.to_array();
    let d = dir.to_array();
    let mn = box_min.to_array();
    let mx = box_max.to_array();

    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;

    for axis in 0..3 {
        if d[axis].abs() < 1e-8 {
            if o[axis] < mn[axis] || o[axis] > mx[axis] {
                return None;
            }
        } else {
            let inv_d = 1.0 / d[axis];
            let mut t1 = (mn[axis] - o[axis]) * inv_d;
            let mut t2 = (mx[axis] - o[axis]) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return None;
            }
        }
    }

    Some((t_min, t_max))
}

/// Casts a ray through the chunk and returns the first solid voxel it hits,
/// using the standard Amanatides & Woo voxel-traversal algorithm.
///
/// `origin` and `dir` are in the same local-chunk coordinate space as the
/// mesher's output (0..CHUNK_SIZE per axis). `dir` need not be normalized.
pub fn raycast_chunk(chunk: &Chunk, origin: Vec3, dir: Vec3) -> Option<RayHit> {
    let size = CHUNK_SIZE as f32;
    let (t_enter, t_exit) = ray_aabb(origin, dir, Vec3::ZERO, Vec3::splat(size))?;

    if t_exit < 0.0 {
        return None;
    }

    let t_start = (t_enter.max(0.0)) + 1e-4;
    let start = origin + dir * t_start;

    let mut cell = IVec3::new(
        (start.x.floor() as i32).clamp(0, CHUNK_SIZE as i32 - 1),
        (start.y.floor() as i32).clamp(0, CHUNK_SIZE as i32 - 1),
        (start.z.floor() as i32).clamp(0, CHUNK_SIZE as i32 - 1),
    );

    let step = IVec3::new(
        if dir.x > 0.0 { 1 } else if dir.x < 0.0 { -1 } else { 0 },
        if dir.y > 0.0 { 1 } else if dir.y < 0.0 { -1 } else { 0 },
        if dir.z > 0.0 { 1 } else if dir.z < 0.0 { -1 } else { 0 },
    );

    let t_delta = Vec3::new(
        if dir.x != 0.0 { (1.0 / dir.x).abs() } else { f32::INFINITY },
        if dir.y != 0.0 { (1.0 / dir.y).abs() } else { f32::INFINITY },
        if dir.z != 0.0 { (1.0 / dir.z).abs() } else { f32::INFINITY },
    );

    let next_boundary = |c: i32, o: f32, d: f32| -> f32 {
        if d > 0.0 {
            ((c + 1) as f32 - o) / d
        } else if d < 0.0 {
            (c as f32 - o) / d
        } else {
            f32::INFINITY
        }
    };

    let mut t_max = Vec3::new(
        next_boundary(cell.x, origin.x, dir.x),
        next_boundary(cell.y, origin.y, dir.y),
        next_boundary(cell.z, origin.z, dir.z),
    );

    let mut prev: Option<IVec3> = None;

    loop {
        if cell.x < 0
            || cell.y < 0
            || cell.z < 0
            || cell.x >= CHUNK_SIZE as i32
            || cell.y >= CHUNK_SIZE as i32
            || cell.z >= CHUNK_SIZE as i32
        {
            return None;
        }

        let voxel = chunk.get(cell.x as usize, cell.y as usize, cell.z as usize);
        if !voxel.is_empty() {
            return Some(RayHit { voxel: cell, place_at: prev });
        }

        prev = Some(cell);

        if t_max.x < t_max.y && t_max.x < t_max.z {
            cell.x += step.x;
            t_max.x += t_delta.x;
        } else if t_max.y < t_max.z {
            cell.y += step.y;
            t_max.y += t_delta.y;
        } else {
            cell.z += step.z;
            t_max.z += t_delta.z;
        }

        if t_max.x.min(t_max.y).min(t_max.z) > t_exit {
            return None;
        }
    }
}