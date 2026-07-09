use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use serde_json::{json, Value};
use crate::MeshData;

/// Generates a deterministic RGB color for a given voxel ID to store in glTF's COLOR_0 attribute.
fn voxel_id_to_color(id: u16) -> [f32; 3] {
    if id == 0 {
        return [1.0, 1.0, 1.0]; // Default white for fallback
    }
    // Simple deterministic hashing to spread colors out nicely 
    let r = ((id * 54) % 256) as f32 / 255.0;
    let g = ((id * 123) % 256) as f32 / 255.0;
    let b = ((id * 211) % 256) as f32 / 255.0;
    [r, g, b]
}

/// Reverse-maps an RGB color back to its closest matching voxel ID.
fn color_to_voxel_id(color: [f32; 3]) -> u16 {
    // Generate a quick reverse lookup table for 1-1000 IDs or look for the minimum distance
    // For a highly performant or large palette, you'd match against an explicit palette map.
    let mut best_id = 1;
    let mut min_dist = f32::MAX;
    
    for id in 1..1000 {
        let c = voxel_id_to_color(id);
        let dist = (c[0] - color[0]).powi(2) + (c[1] - color[1]).powi(2) + (c[2] - color[2]).powi(2);
        if dist < min_dist {
            min_dist = dist;
            best_id = id;
        }
        if min_dist < 1e-5 { break; } // Close enough match
    }
    best_id
}

/// Exports a `MeshData` object into a self-contained Binary glTF (.glb) file.
/// Voxel IDs are backed natively into the vertex stream using the `COLOR_0` attribute.
pub fn export_gltf<P: AsRef<Path>>(mesh: &MeshData, path: P) -> Result<(), Box<dyn std::error::Error>> {
    let vertex_count = mesh.positions.len();
    let index_count = mesh.indices.len();

    // 1. Prepare raw byte buffers
    let mut buffer_bytes = Vec::new();

    // Helper to push elements safely and enforce alignment
    fn push_to_buffer<T: Copy>(buffer: &mut Vec<u8>, data: &[T]) -> (usize, usize) {
        let byte_offset = buffer.len();
        let byte_length = data.len() * std::mem::size_of::<T>();
        let ptr = data.as_ptr() as *const u8;
        let slice = unsafe { std::slice::from_raw_parts(ptr, byte_length) };
        buffer.extend_from_slice(slice);
        // Align to 4 bytes for glTF compliance
        while buffer.len() % 4 != 0 {
            buffer.push(0);
        }
        (byte_offset, byte_length)
    }

    // Write all vertex data arrays into the binary stream
    let (pos_offset, pos_length) = push_to_buffer(&mut buffer_bytes, &mesh.positions);
    let (norm_offset, norm_length) = push_to_buffer(&mut buffer_bytes, &mesh.normals);

    // Generate colors from voxel IDs
    let colors: Vec<[f32; 3]> = mesh.voxel_ids.iter().map(|&id| voxel_id_to_color(id)).collect();
    let (color_offset, color_length) = push_to_buffer(&mut buffer_bytes, &colors);

    // Write index data
    let (index_offset, index_length) = push_to_buffer(&mut buffer_bytes, &mesh.indices);

    // Calculate bounding box for positions
    let mut min_pos = [f32::MAX, f32::MAX, f32::MAX];
    let mut max_pos = [f32::MIN, f32::MIN, f32::MIN];
    for p in &mesh.positions {
        for i in 0..3 {
            if p[i] < min_pos[i] { min_pos[i] = p[i]; }
            if p[i] > max_pos[i] { max_pos[i] = p[i]; }
        }
    }

    // 2. Build the glTF JSON layout structural description
    let gltf_json = json!({
        "asset": { "version": "2.0", "generator": "VoxelEngine" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "COLOR_0": 2
                },
                "indices": 3,
                "mode": 4 // TRIANGLES
            }]
        }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": pos_offset, "byteLength": pos_length, "target": 34962 }, // ARRAY_BUFFER
            { "buffer": 0, "byteOffset": norm_offset, "byteLength": norm_length, "target": 34962 },
            { "buffer": 0, "byteOffset": color_offset, "byteLength": color_length, "target": 34962 },
            { "buffer": 0, "byteOffset": index_offset, "byteLength": index_length, "target": 34963 }  // ELEMENT_ARRAY_BUFFER
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": vertex_count, "type": "VEC3", "min": min_pos, "max": max_pos }, // POSITION (FLOAT)
            { "bufferView": 1, "componentType": 5126, "count": vertex_count, "type": "VEC3" }, // NORMAL (FLOAT)
            { "bufferView": 2, "componentType": 5126, "count": vertex_count, "type": "VEC3" }, // COLOR_0 (FLOAT)
            { "bufferView": 3, "componentType": 5123, "count": index_count, "type": "SCALAR" }  // INDICES (UNSIGNED_SHORT/INT depending on size; u32 is 5125, but we check compatibility or just use 5125 for safety)
        ],
        "buffers": [{ "byteLength": buffer_bytes.len() }]
    });

    // Mutate the component type for indices to 5125 (UNSIGNED_INT) since our input is Vec<u32>
    let mut gltf_json = gltf_json;
    gltf_json["accessors"][3]["componentType"] = json!(5125);

    let json_string = serde_json::to_string(&gltf_json)?;
    let mut json_bytes = json_string.into_bytes();
    
    // JSON chunk must be padded with spaces to a 4-byte boundary
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }

    // 3. Assemble GLB Header and Chunks
    let json_chunk_length = json_bytes.len() as u32;
    let binary_chunk_length = buffer_bytes.len() as u32;
    let total_size = 12 + 8 + json_chunk_length + 8 + binary_chunk_length;

    let mut file = File::create(path)?;
    
    // Header
    file.write_all(b"glTF")?;                  // Magic
    file.write_all(&2u32.to_le_bytes())?;       // Version 2
    file.write_all(&total_size.to_le_bytes())?; // Total file length

    // Chunk 0: JSON
    file.write_all(&json_chunk_length.to_le_bytes())?;
    file.write_all(b"JSON")?;
    file.write_all(&json_bytes)?;

    // Chunk 1: BIN
    file.write_all(&binary_chunk_length.to_le_bytes())?;
    file.write_all(b"BIN\0")?;
    file.write_all(&buffer_bytes)?;

    Ok(())
}

/// Imports and parses a self-contained Binary glTF (.glb) file back into our engine's `MeshData`.
pub fn import_gltf<P: AsRef<Path>>(path: P) -> Result<MeshData, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    
    // Read Header
    let mut header = [0u8; 12];
    file.read_exact(&mut header)?;
    if &header[0..4] != b"glTF" {
        return Err("Invalid glTF file header magic identifier".into());
    }

    // Read JSON Chunk Header
    let mut chunk_header = [0u8; 8];
    file.read_exact(&mut chunk_header)?;
    let json_chunk_len = u32::from_le_bytes([chunk_header[0], chunk_header[1], chunk_header[2], chunk_header[3]]) as usize;
    if &chunk_header[4..8] != b"JSON" {
        return Err("Missing required JSON chunk zero configuration stream".into());
    }

    // Read JSON Content
    let mut json_bytes = vec![0u8; json_chunk_len];
    file.read_exact(&mut json_bytes)?;
    let gltf: Value = serde_json::from_slice(&json_bytes)?;

    // Read BIN Chunk Header
    file.read_exact(&mut chunk_header)?;
    let bin_chunk_len = u32::from_le_bytes([chunk_header[0], chunk_header[1], chunk_header[2], chunk_header[3]]) as usize;
    if &chunk_header[4..8] != b"BIN\0" {
        return Err("Missing required BIN layout binary data stream".into());
    }

    // Read Binary Content
    let mut bin_bytes = vec![0u8; bin_chunk_len];
    file.read_exact(&mut bin_bytes)?;

    // Parse primitive accessors targeting vertex arrays
    let primitive = &gltf["meshes"][0]["primitives"][0];
    let pos_accessor_idx = primitive["attributes"]["POSITION"].as_u64().unwrap() as usize;
    let norm_accessor_idx = primitive["attributes"]["NORMAL"].as_u64().unwrap() as usize;
    let color_accessor_idx = primitive["attributes"]["COLOR_0"].as_u64().unwrap() as usize;
    let indices_accessor_idx = primitive["indices"].as_u64().unwrap() as usize;

    // Helper closure to read slice from raw byte offsets
    let get_buffer_slice = |accessor_idx: usize| -> &[u8] {
        let accessor = &gltf["accessors"][accessor_idx];
        let view_idx = accessor["bufferView"].as_u64().unwrap() as usize;
        let view = &gltf["bufferViews"][view_idx];
        
        let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize;
        let length = view["byteLength"].as_u64().unwrap() as usize;
        &bin_bytes[offset..(offset + length)]
    };

    // Cast raw bytes safely to internal typed elements
    let pos_slice = get_buffer_slice(pos_accessor_idx);
    let positions: Vec<[f32; 3]> = pos_slice.chunks_exact(12)
        .map(|c| [f32::from_le_bytes(c[0..4].try_into().unwrap()), f32::from_le_bytes(c[4..8].try_into().unwrap()), f32::from_le_bytes(c[8..12].try_into().unwrap())])
        .collect();

    let norm_slice = get_buffer_slice(norm_accessor_idx);
    let normals: Vec<[f32; 3]> = norm_slice.chunks_exact(12)
        .map(|c| [f32::from_le_bytes(c[0..4].try_into().unwrap()), f32::from_le_bytes(c[4..8].try_into().unwrap()), f32::from_le_bytes(c[8..12].try_into().unwrap())])
        .collect();

    let color_slice = get_buffer_slice(color_accessor_idx);
    let voxel_ids: Vec<u16> = color_slice.chunks_exact(12)
        .map(|c| {
            let rgb = [f32::from_le_bytes(c[0..4].try_into().unwrap()), f32::from_le_bytes(c[4..8].try_into().unwrap()), f32::from_le_bytes(c[8..12].try_into().unwrap())];
            color_to_voxel_id(rgb)
        })
        .collect();

    let index_slice = get_buffer_slice(indices_accessor_idx);
    let indices: Vec<u32> = index_slice.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    Ok(MeshData {
        positions,
        normals,
        voxel_ids,
        indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gltf_roundtrip() {
        // Construct mock input data matching full properties of mesh structures
        let source_mesh = MeshData {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            voxel_ids: vec![42, 128, 512],
            indices: vec![0, 1, 2],
        };

        let path = "test_roundtrip.glb";
        
        // Export 
        export_gltf(&source_mesh, path).expect("Failed to export GLB file");

        // Import back again
        let parsed_mesh = import_gltf(path).expect("Failed to import GLB file back");

        // Cleanup temporary file
        let _ = std::fs::remove_file(path);

        // Validation Assertions
        assert_eq!(source_mesh.positions, parsed_mesh.positions);
        assert_eq!(source_mesh.normals, parsed_mesh.normals);
        assert_eq!(source_mesh.voxel_ids, parsed_mesh.voxel_ids);
        assert_eq!(source_mesh.indices, parsed_mesh.indices);
    }
}