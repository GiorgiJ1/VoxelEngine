#[derive(Debug, Default, Clone)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub voxel_ids: Vec<u16>,
    pub indices: Vec<u32>,
}

impl MeshData {
    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}