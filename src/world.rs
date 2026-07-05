use std::collections::HashMap;

use glam::IVec3;

use crate::chunk::{Chunk, CHUNK_SIZE};
use crate::voxel::Voxel;

pub struct VoxelWorld {
    chunks: HashMap<IVec3, Chunk>,
}

impl VoxelWorld {

    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    fn world_to_chunk_local(pos: IVec3) -> (IVec3, (usize, usize, usize)) {
        let size = CHUNK_SIZE as i32;
        let chunk_pos = IVec3::new(
            pos.x.div_euclid(size),
            pos.y.div_euclid(size),
            pos.z.div_euclid(size),
        );
        let local = IVec3::new(
            pos.x.rem_euclid(size),
            pos.y.rem_euclid(size),
            pos.z.rem_euclid(size),
        );
        (chunk_pos, (local.x as usize, local.y as usize, local.z as usize))
    }

    pub fn get_voxel(&self, pos: IVec3) -> Voxel {
        let (chunk_pos, (lx, ly, lz)) = Self::world_to_chunk_local(pos);
        self.chunks
            .get(&chunk_pos)
            .map(|chunk| chunk.get(lx, ly, lz))
            .unwrap_or(Voxel::EMPTY)
    }

    pub fn set_voxel(&mut self, pos: IVec3, voxel: Voxel) {
        let (chunk_pos, (lx, ly, lz)) = Self::world_to_chunk_local(pos);
        let chunk = self.chunks.entry(chunk_pos).or_insert_with(Chunk::empty);
        chunk.set(lx, ly, lz, voxel);
    }
    pub fn chunk(&self, chunk_pos: IVec3) -> Option<&Chunk> {
        self.chunks.get(&chunk_pos)
    }
    pub fn iter_chunks(&self) -> impl Iterator<Item = (&IVec3, &Chunk)> {
        self.chunks.iter()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self::new()
    }
}