#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Voxel(pub u16);

impl Voxel {
    pub const EMPTY: Voxel = Voxel(0);

    pub fn new(id: u16) -> Self {
        Voxel(id)
    }

    pub fn is_empty(&self) -> bool {
        self.0 == Self::EMPTY.0
    }
}