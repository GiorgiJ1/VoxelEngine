use crate::voxel::Voxel;

/// Size of a chunk along each axis. 16 is a common sweet spot:
/// small enough to remesh quickly when edited, big enough to keep
/// chunk-count overhead low.
pub const CHUNK_SIZE: usize = 16;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

/// A fixed-size 16x16x16 block of voxels. Dense storage (not sparse) --
/// simplest possible thing that works. We can swap the internals for
/// something smarter (RLE, palette-compressed) later without touching
/// any calling code, since `get`/`set` are the only public interface.
pub struct Chunk {
    voxels: Box<[Voxel; CHUNK_VOLUME]>,
}

impl Chunk {
    /// A brand new chunk, entirely air.
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

    /// Local-space get. x/y/z must each be in 0..CHUNK_SIZE.
    pub fn get(&self, x: usize, y: usize, z: usize) -> Voxel {
        self.voxels[Self::index(x, y, z)]
    }

    /// Local-space set. x/y/z must each be in 0..CHUNK_SIZE.
    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel: Voxel) {
        self.voxels[Self::index(x, y, z)] = voxel;
    }

    /// True if every voxel in this chunk is empty. Useful later so the
    /// mesher and the world can skip fully-air chunks entirely.
    pub fn is_empty(&self) -> bool {
        self.voxels.iter().all(|v| v.is_empty())
    }

    /// Flattens this chunk's voxels into a plain Vec of raw ids, in the
    /// same x + y*SIZE + z*SIZE*SIZE order as internal storage. Used for
    /// serialization -- a Vec<u16> is trivially (de)serializable with serde
    /// regardless of CHUNK_SIZE, unlike a fixed-size array, which sidesteps
    /// needing a big-array crate just to save a chunk to disk.
    pub fn to_ids(&self) -> Vec<u16> {
        self.voxels.iter().map(|v| v.0).collect()
    }

    /// Rebuilds a chunk from a flat id list produced by `to_ids`. Returns
    /// `None` if the length doesn't match CHUNK_VOLUME (e.g. a save file
    /// from a different CHUNK_SIZE) rather than panicking on bad input.
    pub fn from_ids(ids: &[u16]) -> Option<Self> {
        if ids.len() != CHUNK_VOLUME {
            return None;
        }
        let mut voxels = Box::new([Voxel::EMPTY; CHUNK_VOLUME]);
        for (slot, id) in voxels.iter_mut().zip(ids.iter()) {
            *slot = Voxel::new(*id);
        }
        Some(Self { voxels })
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self::empty()
    }
}