use glam::IVec3;

pub fn line_voxels(start: IVec3, end: IVec3) -> Vec<IVec3> {
    let mut points = Vec::new();

    let (mut x, mut y, mut z) = (start.x, start.y, start.z);
    let (x1, y1, z1) = (end.x, end.y, end.z);

    let dx = (x1 - x).abs();
    let dy = (y1 - y).abs();
    let dz = (z1 - z).abs();

    let sx = if x1 > x { 1 } else { -1 };
    let sy = if y1 > y { 1 } else { -1 };
    let sz = if z1 > z { 1 } else { -1 };

    if dx >= dy && dx >= dz {
        let mut p1 = 2 * dy - dx;
        let mut p2 = 2 * dz - dx;
        while x != x1 {
            points.push(IVec3::new(x, y, z));
            if p1 >= 0 {
                y += sy;
                p1 -= 2 * dx;
            }
            if p2 >= 0 {
                z += sz;
                p2 -= 2 * dx;
            }
            p1 += 2 * dy;
            p2 += 2 * dz;
            x += sx;
        }
    } else if dy >= dx && dy >= dz {
        let mut p1 = 2 * dx - dy;
        let mut p2 = 2 * dz - dy;
        while y != y1 {
            points.push(IVec3::new(x, y, z));
            if p1 >= 0 {
                x += sx;
                p1 -= 2 * dy;
            }
            if p2 >= 0 {
                z += sz;
                p2 -= 2 * dy;
            }
            p1 += 2 * dx;
            p2 += 2 * dz;
            y += sy;
        }
    } else {
        let mut p1 = 2 * dy - dz;
        let mut p2 = 2 * dx - dz;
        while z != z1 {
            points.push(IVec3::new(x, y, z));
            if p1 >= 0 {
                y += sy;
                p1 -= 2 * dz;
            }
            if p2 >= 0 {
                x += sx;
                p2 -= 2 * dz;
            }
            p1 += 2 * dy;
            p2 += 2 * dx;
            z += sz;
        }
    }

    points.push(IVec3::new(x1, y1, z1));
    points
}