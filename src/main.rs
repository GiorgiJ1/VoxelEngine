use glam::IVec3;
use voxel_core::{Voxel, VoxelWorld};

fn main() {
    let mut world = VoxelWorld::new();

    for x in -2..1 {
        for y in 0..3 {
            for z in 0..3 {
                world.set_voxel(IVec3::new(x, y, z), Voxel::new(1));
            }
        }
    }

    assert_eq!(world.get_voxel(IVec3::new(-1, 1, 1)), Voxel::new(1));
    assert_eq!(world.get_voxel(IVec3::new(100, 100, 100)), Voxel::EMPTY);

    println!("chunks touched: {}", world.chunk_count());
    println!("voxel at (-1,1,1): {:?}", world.get_voxel(IVec3::new(-1, 1, 1)));
    println!("voxel at (5,5,5):  {:?}", world.get_voxel(IVec3::new(5, 5, 5)));
    println!("all good ✅");
}