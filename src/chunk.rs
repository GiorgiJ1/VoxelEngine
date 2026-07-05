use crate::voxel::Voxel;

pub const CHUNK_SIZE: usize = 16;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

pub struct Chunk {
    voxels: Box<[Voxel; CHUNK_VOLUME]>,
}

impl Chunk {

    pub fn empty() -> Self {
        Self {
            voxels: Box::new([Voxel::EMPTY; CHUNK_VOLUME]),
        }
    }

    #[inline]
    fn index(x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
        x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
    }
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        self.voxels[Self::index(x, y, z)]
    }
    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        self.voxels[Self::index(x, y, z)] = voxel;
    }
    pub fn is_empty(&self) -> bool {
        self.voxels.iter().all(|v| v.is_empty())
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::empty()
    }
}