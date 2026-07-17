use glam::{IVec3, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl Face {

    pub fn from_normal(n: IVec3) -> Option<Face> {
        match (n.x, n.y, n.z) {
            (1, 0, 0) => Some(Face::PosX),
            (-1, 0, 0) => Some(Face::NegX),
            (0, 1, 0) => Some(Face::PosY),
            (0, -1, 0) => Some(Face::NegY),
            (0, 0, 1) => Some(Face::PosZ),
            (0, 0, -1) => Some(Face::NegZ),
            _ => None,
        }
    }

    pub fn normal(self) -> IVec3 {
        match self {
            Face::PosX => IVec3::new(1, 0, 0),
            Face::NegX => IVec3::new(-1, 0, 0),
            Face::PosY => IVec3::new(0, 1, 0),
            Face::NegY => IVec3::new(0, -1, 0),
            Face::PosZ => IVec3::new(0, 0, 1),
            Face::NegZ => IVec3::new(0, 0, -1),
        }
    }

    pub fn normal_f32(self) -> Vec3 {
        let n = self.normal();
        Vec3::new(n.x as f32, n.y as f32, n.z as f32)
    }

    pub fn locked_axis(self) -> usize {
        match self {
            Face::PosX | Face::NegX => 0,
            Face::PosY | Face::NegY => 1,
            Face::PosZ | Face::NegZ => 2,
        }
    }

    pub fn in_plane_axes(self) -> (usize, usize) {
        match self.locked_axis() {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        }
    }
}