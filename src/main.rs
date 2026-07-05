use voxel_core::{greedy_mesh, Chunk, Voxel};

fn main() {
    let mut chunk = Chunk::empty();
    chunk.set(0, 0, 0, Voxel::new(1));
    let mesh = greedy_mesh(&chunk);
    println!(
        "single voxel  -> quads: {}, verts: {}, tris: {}",
        mesh.quad_count(),
        mesh.positions.len(),
        mesh.triangle_count()
    );
    assert_eq!(mesh.quad_count(), 6);
    assert_eq!(mesh.positions.len(), 24);

    let mut full = Chunk::empty();
    for x in 0..16 {
        for y in 0..16 {
            for z in 0..16 {
                full.set(x, y, z, Voxel::new(1));
            }
        }
    }
    let mesh_full = greedy_mesh(&full);
    println!(
        "solid chunk   -> quads: {}, verts: {}, tris: {}",
        mesh_full.quad_count(),
        mesh_full.positions.len(),
        mesh_full.triangle_count()
    );
    assert_eq!(mesh_full.quad_count(), 6);

    let mut slab = Chunk::empty();
    for x in 0..16 {
        for z in 0..16 {
            slab.set(x, 0, z, Voxel::new(2));
            slab.set(x, 1, z, Voxel::new(2));
        }
    }
    let mesh_slab = greedy_mesh(&slab);
    println!(
        "2-tall slab   -> quads: {}, verts: {}, tris: {}",
        mesh_slab.quad_count(),
        mesh_slab.positions.len(),
        mesh_slab.triangle_count()
    );
    assert_eq!(mesh_slab.quad_count(), 6);

    println!("all mesher checks passed ✅");
}