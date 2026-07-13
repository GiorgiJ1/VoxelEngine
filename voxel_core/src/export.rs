use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use crate::mesh::MeshData;

// --- Minimal glTF 2.0 JSON schema ---

#[derive(Serialize, Deserialize)]
struct GltfAsset {
    version: String,
}

#[derive(Serialize, Deserialize)]
struct GltfBuffer {
    #[serde(rename = "byteLength")]
    byte_length: usize,
}

#[derive(Serialize, Deserialize)]
struct GltfBufferView {
    buffer: u32,
    #[serde(rename = "byteOffset")]
    byte_offset: usize,
    #[serde(rename = "byteLength")]
    byte_length: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<u32>,
}

#[derive(Serialize, Deserialize)]
struct GltfAccessor {
    #[serde(rename = "bufferView")]
    buffer_view: u32,
    #[serde(rename = "componentType")]
    component_type: u32,
    count: usize,
    #[serde(rename = "type")]
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
struct GltfMeshPrimitive {
    attributes: GltfAttributes,
    indices: u32,
    mode: u32,
}

#[derive(Serialize, Deserialize)]
struct GltfAttributes {
    #[serde(rename = "POSITION")]
    position: u32,
    #[serde(rename = "NORMAL")]
    normal: u32,
    #[serde(rename = "COLOR_0")]
    color_0: u32,
}

#[derive(Serialize, Deserialize)]
struct GltfMesh {
    primitives: Vec<GltfMeshPrimitive>,
}

#[derive(Serialize, Deserialize)]
struct GltfNode {
    mesh: u32,
}

#[derive(Serialize, Deserialize)]
struct GltfScene {
    nodes: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct GltfRoot {
    asset: GltfAsset,
    buffers: Vec<GltfBuffer>,
    #[serde(rename = "bufferViews")]
    buffer_views: Vec<GltfBufferView>,
    accessors: Vec<GltfAccessor>,
    meshes: Vec<GltfMesh>,
    nodes: Vec<GltfNode>,
    scenes: Vec<GltfScene>,
    scene: u32,
}

pub fn export_gltf_glb(mesh: &MeshData, path: &Path, materials_resolver: &[ [f32; 3]; 256 ]) -> io::Result<()> {
    let mut pos_bytes = Vec::new();
    let mut norm_bytes = Vec::new();
    let mut col_bytes = Vec::new();
    let mut idx_bytes = Vec::new();

    let mut min_pos = [f32::INFINITY; 3];
    let mut max_pos = [f32::NEG_INFINITY; 3];

    for &p in &mesh.positions {
        for axis in 0..3 {
            min_pos[axis] = min_pos[axis].min(p[axis]);
            max_pos[axis] = max_pos[axis].max(p[axis]);
        }
        pos_bytes.extend_from_slice(bytemuck::bytes_of(&p));
    }

    for &n in &mesh.normals {
        norm_bytes.extend_from_slice(bytemuck::bytes_of(&n));
    }

    for &voxel_id in &mesh.voxel_ids {
        let rgb = materials_resolver.get(voxel_id as usize).copied().unwrap_or([0.5, 0.5, 0.5]);
        let rgba = [rgb[0], rgb[1], rgb[2], 1.0f32];
        col_bytes.extend_from_slice(bytemuck::bytes_of(&rgba));
    }

    for &idx in &mesh.indices {
        idx_bytes.extend_from_slice(bytemuck::bytes_of(&idx));
    }

    let mut bin_buffer = Vec::new();
    let view_pos_offset = bin_buffer.len();
    bin_buffer.extend_from_slice(&pos_bytes);
    let view_pos_len = pos_bytes.len();

    let view_norm_offset = bin_buffer.len();
    bin_buffer.extend_from_slice(&norm_bytes);
    let view_norm_len = norm_bytes.len();

    let view_col_offset = bin_buffer.len();
    bin_buffer.extend_from_slice(&col_bytes);
    let view_col_len = col_bytes.len();

    let view_idx_offset = bin_buffer.len();
    bin_buffer.extend_from_slice(&idx_bytes);
    let view_idx_len = idx_bytes.len();

    while bin_buffer.len() % 4 != 0 {
        bin_buffer.push(0);
    }

    let root = GltfRoot {
        asset: GltfAsset { version: "2.0".to_string() },
        buffers: vec![GltfBuffer { byte_length: bin_buffer.len() }],
        buffer_views: vec![
            GltfBufferView { buffer: 0, byte_offset: view_pos_offset, byte_length: view_pos_len, target: Some(34962) },
            GltfBufferView { buffer: 0, byte_offset: view_norm_offset, byte_length: view_norm_len, target: Some(34962) },
            GltfBufferView { buffer: 0, byte_offset: view_col_offset, byte_length: view_col_len, target: Some(34962) },
            GltfBufferView { buffer: 0, byte_offset: view_idx_offset, byte_length: view_idx_len, target: Some(34963) },
        ],
        accessors: vec![
            GltfAccessor {
                buffer_view: 0,
                component_type: 5126,
                count: mesh.positions.len(),
                kind: "VEC3".to_string(),
                min: Some(min_pos.to_vec()),
                max: Some(max_pos.to_vec()),
            },
            GltfAccessor {
                buffer_view: 1,
                component_type: 5126,
                count: mesh.normals.len(),
                kind: "VEC3".to_string(),
                min: None,
                max: None,
            },
            GltfAccessor {
                buffer_view: 2,
                component_type: 5126,
                count: mesh.voxel_ids.len(),
                kind: "VEC4".to_string(),
                min: None,
                max: None,
            },
            GltfAccessor {
                buffer_view: 3,
                component_type: 5125,
                count: mesh.indices.len(),
                kind: "SCALAR".to_string(),
                min: None,
                max: None,
            },
        ],
        meshes: vec![GltfMesh {
            primitives: vec![GltfMeshPrimitive {
                attributes: GltfAttributes { position: 0, normal: 1, color_0: 2 },
                indices: 3,
                mode: 4,
            }],
        }],
        nodes: vec![GltfNode { mesh: 0 }],
        scenes: vec![GltfScene { nodes: vec![0] }],
        scene: 0,
    };

    let json_str = serde_json::to_string(&root).expect("serialize gltf json");
    let mut json_bytes = json_str.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(0x20);
    }

    let total_size = 12 + 8 + json_bytes.len() + 8 + bin_buffer.len();
    let mut out = Vec::with_capacity(total_size);

    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total_size as u32).to_le_bytes());

    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);

    out.extend_from_slice(&(bin_buffer.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin_buffer);

    fs::write(path, out)?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ImportedMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

pub fn import_gltf_glb(path: &Path) -> Result<ImportedMesh, String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read file: {e}"))?;
    if bytes.len() < 20 {
        return Err("file too short to be a valid GLB".to_string());
    }
    if &bytes[0..4] != b"glTF" {
        return Err("magic bytes are not glTF".to_string());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != 2 {
        return Err(format!("unsupported glTF version: {version}"));
    }

    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json_type = &bytes[16..20];
    if json_type != b"JSON" {
        return Err("first chunk is not JSON".to_string());
    }

    let json_start = 20;
    let json_end = json_start + json_len;
    if json_end > bytes.len() {
        return Err("JSON chunk extends past file boundary".to_string());
    }
    let json_slice = &bytes[json_start..json_end];
    let root: GltfRoot = serde_json::from_slice(json_slice).map_err(|e| format!("failed to parse JSON: {e}"))?;

    let bin_header_start = json_end;
    if bin_header_start + 8 > bytes.len() {
        return Err("missing BIN chunk header".to_string());
    }
    let bin_len = u32::from_le_bytes(bytes[bin_header_start..bin_header_start + 4].try_into().unwrap()) as usize;
    let bin_type = &bytes[bin_header_start + 4..bin_header_start + 8];
    if bin_type != b"BIN\0" {
        return Err("second chunk is not BIN".to_string());
    }

    let bin_start = bin_header_start + 8;
    let bin_end = bin_start + bin_len;
    if bin_end > bytes.len() {
        return Err("BIN chunk extends past file boundary".to_string());
    }
    let bin_bytes = &bytes[bin_start..bin_end];

    if root.meshes.is_empty() || root.meshes[0].primitives.is_empty() {
        return Err("no mesh primitives found in GLB".to_string());
    }
    let primitive = &root.meshes[0].primitives[0];

    let read_f32 = |buf: &[u8], offset: usize| -> f32 {
        f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    };
    let read_u32 = |buf: &[u8], offset: usize| -> u32 {
        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
    };

    let read_vec_accessor = |accessor_idx: u32, components: usize| -> Result<Vec<f32>, String> {
        let accessor = root.accessors.get(accessor_idx as usize).ok_or("accessor out of range")?;
        let view = root.buffer_views.get(accessor.buffer_view as usize).ok_or("bufferView out of range")?;
        let mut out = Vec::with_capacity(accessor.count * components);
        for i in 0..accessor.count {
            for c in 0..components {
                let offset = view.byte_offset + (i * components + c) * 4;
                out.push(read_f32(bin_bytes, offset));
            }
        }
        Ok(out)
    };

    let flat_positions = read_vec_accessor(primitive.attributes.position, 3)?;
    let flat_normals = read_vec_accessor(primitive.attributes.normal, 3)?;
    let flat_colors = read_vec_accessor(primitive.attributes.color_0, 4)?;

    let indices_accessor = root.accessors.get(primitive.indices as usize).ok_or("indices accessor out of range")?;
    let indices_view = root.buffer_views.get(indices_accessor.buffer_view as usize).ok_or("indices bufferView out of range")?;
    let mut indices = Vec::with_capacity(indices_accessor.count);
    for i in 0..indices_accessor.count {
        indices.push(read_u32(bin_bytes, indices_view.byte_offset + i * 4));
    }

    let positions = flat_positions.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let normals = flat_normals.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let colors = flat_colors.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();

    Ok(ImportedMesh { positions, normals, colors, indices })
}

/// Exports the mesh to a classic split .obj + .mtl pair.
///
/// Takes the same `materials_resolver` shape as `export_gltf_glb` (rather
/// than owning its own copy of the palette) and only emits materials that
/// are actually present in `mesh.voxel_ids` -- derived from the mesh
/// itself, not a fixed list. That matters: a hardcoded list of known ids
/// would silently drop geometry for any material id it didn't know about
/// (e.g. if the palette ever grows past whatever was hardcoded here).
pub fn export_obj_mtl(mesh: &MeshData, base_path: &Path, materials_resolver: &[[f32; 3]; 256]) -> io::Result<()> {
    let obj_path = base_path.with_extension("obj");
    let mtl_path = base_path.with_extension("mtl");
    let mtl_filename = mtl_path.file_name().unwrap().to_string_lossy();

    let mut used_ids: Vec<u16> = mesh
        .voxel_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    used_ids.sort_unstable();

    // 1. Write the .mtl file, one entry per material actually used.
    let mut mtl_file = File::create(&mtl_path)?;
    writeln!(mtl_file, "# Generated Material Library")?;
    for id in &used_ids {
        let rgb = materials_resolver[*id as usize];
        writeln!(mtl_file, "newmtl VoxelMat_{id}")?;
        writeln!(mtl_file, "Kd {} {} {}", rgb[0], rgb[1], rgb[2])?;
        writeln!(mtl_file, "illum 1")?;
        writeln!(mtl_file, "Ka 0.2 0.2 0.2")?;
        writeln!(mtl_file, "Ks 0.0 0.0 0.0")?;
        writeln!(mtl_file)?;
    }

    // 2. Write the .obj geometry, grouped by material.
    let mut obj_file = File::create(&obj_path)?;
    writeln!(obj_file, "# Voxel Engine Export")?;
    writeln!(obj_file, "mtllib {mtl_filename}")?;
    writeln!(obj_file)?;

    for pos in &mesh.positions {
        writeln!(obj_file, "v {} {} {}", pos[0], pos[1], pos[2])?;
    }
    writeln!(obj_file)?;

    for norm in &mesh.normals {
        writeln!(obj_file, "vn {} {} {}", norm[0], norm[1], norm[2])?;
    }
    writeln!(obj_file)?;

    let quad_count = mesh.indices.len() / 6;
    for id in &used_ids {
        let mut written_mat_header = false;

        for q in 0..quad_count {
            let i0 = mesh.indices[q * 6] as usize;
            let current_quad_mat = mesh.voxel_ids.get(i0).copied().unwrap_or(0);

            if current_quad_mat == *id {
                if !written_mat_header {
                    writeln!(obj_file, "usemtl VoxelMat_{id}")?;
                    written_mat_header = true;
                }

                let t1_0 = mesh.indices[q * 6] + 1;
                let t1_1 = mesh.indices[q * 6 + 1] + 1;
                let t1_2 = mesh.indices[q * 6 + 2] + 1;
                writeln!(obj_file, "f {}//{} {}//{} {}//{}", t1_0, t1_0, t1_1, t1_1, t1_2, t1_2)?;

                let t2_0 = mesh.indices[q * 6 + 3] + 1;
                let t2_1 = mesh.indices[q * 6 + 4] + 1;
                let t2_2 = mesh.indices[q * 6 + 5] + 1;
                writeln!(obj_file, "f {}//{} {}//{} {}//{}", t2_0, t2_0, t2_1, t2_1, t2_2, t2_2)?;
            }
        }
    }

    println!("Exported OBJ asset pair successfully.");
    Ok(())
}