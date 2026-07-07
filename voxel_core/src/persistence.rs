use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::chunk::Chunk;

/// On-disk representation of a saved chunk. Kept deliberately tiny and
/// separate from `Chunk` itself -- `Chunk`'s internal storage can change
/// (e.g. to something sparse) without breaking the save format, since this
/// struct is the only thing that actually gets (de)serialized.
#[derive(Serialize, Deserialize)]
struct ChunkFile {
    /// Bumped if the save format ever changes shape, so `load_chunk` can
    /// give a clear error instead of silently misreading old files.
    version: u32,
    ids: Vec<u16>,
}

const CURRENT_VERSION: u32 = 1;

/// Errors that can happen loading a save file, distinct from plain I/O
/// errors so the caller (and the UI, eventually) can tell "file doesn't
/// exist yet" apart from "file exists but is corrupt/wrong shape".
#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    Decode(bincode::Error),
    UnsupportedVersion(u32),
    WrongVoxelCount,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "I/O error: {e}"),
            LoadError::Decode(e) => write!(f, "couldn't decode save file: {e}"),
            LoadError::UnsupportedVersion(v) => write!(f, "save file version {v} is not supported"),
            LoadError::WrongVoxelCount => write!(f, "save file's voxel count doesn't match this build's CHUNK_SIZE"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Saves a chunk to `path` as a small binary file.
pub fn save_chunk(chunk: &Chunk, path: &Path) -> io::Result<()> {
    let file = ChunkFile { version: CURRENT_VERSION, ids: chunk.to_ids() };
    let bytes = bincode::serialize(&file).expect("in-memory serialization should never fail");
    fs::write(path, bytes)
}

/// Loads a chunk previously written by `save_chunk`.
pub fn load_chunk(path: &Path) -> Result<Chunk, LoadError> {
    let bytes = fs::read(path).map_err(LoadError::Io)?;
    let file: ChunkFile = bincode::deserialize(&bytes).map_err(LoadError::Decode)?;
    if file.version != CURRENT_VERSION {
        return Err(LoadError::UnsupportedVersion(file.version));
    }
    Chunk::from_ids(&file.ids).ok_or(LoadError::WrongVoxelCount)
}