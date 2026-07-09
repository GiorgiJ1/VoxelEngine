use serde::{Deserialize, Serialize};

use crate::mesh::MeshData;

// --- Minimal glTF 2.0 JSON schema (just the subset we actually emit) ---

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
struct GltfAttributes {
    #[serde(rename = "POSITION")]
    position: u32,
    #[serde(rename = "NORMAL")]
    normal: u32,
    #[serde(rename = "COLOR_0")]
    color_0: u32,
}

#[derive(Serialize, Deserialize)]
struct GltfPrimitive {
    attributes: GltfAttributes,
    indices: u32,
    mode: u32,
}

#[derive(Serialize, Deserialize)]
struct GltfMesh {
    primitives: Vec<GltfPrimitive>,
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
    scene: u32,
    scenes: Vec<GltfScene>,
    nodes: Vec<GltfNode>,
    meshes: Vec<GltfMesh>,
    accessors: Vec<GltfAccessor>,
    #[serde(rename = "bufferViews")]
    buffer_views: Vec<GltfBufferView>,
    buffers: Vec<GltfBuffer>,
}

const COMPONENT_TYPE_FLOAT: u32 = 5126;
const COMPONENT_TYPE_UNSIGNED_INT: u32 = 5125;
const MODE_TRIANGLES: u32 = 4;
const TARGET_ARRAY_BUFFER: u32 = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;

fn positions_min_max(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in positions {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    if positions.is_empty() {
        min = [0.0; 3];
        max = [0.0; 3];
    }
    (min, max)
}

/// Exports a mesh as a self-contained binary glTF (.glb) file.
///
/// `color_for_id` maps a voxel material id to an RGB color -- the caller
/// owns the actual palette (names, ordering, etc.), this function only
/// needs the colors themselves. Colors are baked in as a per-vertex
/// COLOR_0 attribute rather than separate materials/textures, which is the
/// simplest way to carry voxel-art coloring into glTF; most viewers
/// (Blender, three.js, Unity's glTFast) honor vertex colors automatically.
/// Unreal's importer sometimes needs the material graph adjusted to
/// multiply vertex color in -- it isn't always automatic there.
pub fn export_gltf_glb(mesh: &MeshData, color_for_id: impl Fn(u16) -> [f32; 3]) -> Vec<u8> {
    let vertex_count = mesh.positions.len();

    // Lay out the binary buffer as four contiguous, already-4-byte-aligned
    // sections (every component here is either 4-byte f32 or 4-byte u32,
    // so no manual padding is needed between sections).
    let mut bin = Vec::with_capacity(vertex_count * (12 + 12 + 16) + mesh.indices.len() * 4);

    let positions_offset = bin.len();
    for p in &mesh.positions {
        for component in p {
            bin.extend_from_slice(&component.to_le_bytes());
        }
    }
    let positions_len = bin.len() - positions_offset;

    let normals_offset = bin.len();
    for n in &mesh.normals {
        for component in n {
            bin.extend_from_slice(&component.to_le_bytes());
        }
    }
    let normals_len = bin.len() - normals_offset;

    let colors_offset = bin.len();
    for id in &mesh.voxel_ids {
        let [r, g, b] = color_for_id(*id);
        for component in [r, g, b, 1.0] {
            bin.extend_from_slice(&component.to_le_bytes());
        }
    }
    let colors_len = bin.len() - colors_offset;

    let indices_offset = bin.len();
    for i in &mesh.indices {
        bin.extend_from_slice(&i.to_le_bytes());
    }
    let indices_len = bin.len() - indices_offset;

    let (min, max) = positions_min_max(&mesh.positions);

    let root = GltfRoot {
        asset: GltfAsset { version: "2.0".to_string() },
        scene: 0,
        scenes: vec![GltfScene { nodes: vec![0] }],
        nodes: vec![GltfNode { mesh: 0 }],
        meshes: vec![GltfMesh {
            primitives: vec![GltfPrimitive {
                attributes: GltfAttributes { position: 0, normal: 1, color_0: 2 },
                indices: 3,
                mode: MODE_TRIANGLES,
            }],
        }],
        accessors: vec![
            GltfAccessor {
                buffer_view: 0,
                component_type: COMPONENT_TYPE_FLOAT,
                count: vertex_count,
                kind: "VEC3".to_string(),
                min: Some(min.to_vec()),
                max: Some(max.to_vec()),
            },
            GltfAccessor {
                buffer_view: 1,
                component_type: COMPONENT_TYPE_FLOAT,
                count: vertex_count,
                kind: "VEC3".to_string(),
                min: None,
                max: None,
            },
            GltfAccessor {
                buffer_view: 2,
                component_type: COMPONENT_TYPE_FLOAT,
                count: vertex_count,
                kind: "VEC4".to_string(),
                min: None,
                max: None,
            },
            GltfAccessor {
                buffer_view: 3,
                component_type: COMPONENT_TYPE_UNSIGNED_INT,
                count: mesh.indices.len(),
                kind: "SCALAR".to_string(),
                min: None,
                max: None,
            },
        ],
        buffer_views: vec![
            GltfBufferView {
                buffer: 0,
                byte_offset: positions_offset,
                byte_length: positions_len,
                target: Some(TARGET_ARRAY_BUFFER),
            },
            GltfBufferView {
                buffer: 0,
                byte_offset: normals_offset,
                byte_length: normals_len,
                target: Some(TARGET_ARRAY_BUFFER),
            },
            GltfBufferView {
                buffer: 0,
                byte_offset: colors_offset,
                byte_length: colors_len,
                target: Some(TARGET_ARRAY_BUFFER),
            },
            GltfBufferView {
                buffer: 0,
                byte_offset: indices_offset,
                byte_length: indices_len,
                target: Some(TARGET_ELEMENT_ARRAY_BUFFER),
            },
        ],
        buffers: vec![GltfBuffer { byte_length: bin.len() }],
    };

    let mut json = serde_json::to_vec(&root).expect("glTF JSON structure should always serialize");
    while json.len() % 4 != 0 {
        json.push(b' '); // GLB spec: JSON chunk is padded with spaces
    }
    while bin.len() % 4 != 0 {
        bin.push(0); // GLB spec: BIN chunk is padded with zero bytes
    }

    let total_len = 12 + 8 + json.len() + 8 + bin.len();

    let mut glb = Vec::with_capacity(total_len);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&(total_len as u32).to_le_bytes());

    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json);

    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);

    glb
}

/// A mesh read back from a .glb file produced by `export_gltf_glb`.
///
/// This is a round-trip reader for *our own* exporter's output, not a
/// general-purpose glTF importer -- it assumes exactly one buffer, one
/// mesh, one primitive with POSITION/NORMAL/COLOR_0/indices, matching what
/// `export_gltf_glb` always produces.
pub struct ImportedMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

pub fn import_gltf_glb(glb: &[u8]) -> Result<ImportedMesh, String> {
    if glb.len() < 12 || &glb[0..4] != b"glTF" {
        return Err("not a .glb file (bad magic)".to_string());
    }
    let total_len = read_u32(glb, 8) as usize;
    if total_len != glb.len() {
        return Err(format!("header length {total_len} doesn't match actual file size {}", glb.len()));
    }

    let mut cursor = 12;
    let mut json_bytes: Option<&[u8]> = None;
    let mut bin_bytes: Option<&[u8]> = None;

    while cursor + 8 <= glb.len() {
        let chunk_len = read_u32(glb, cursor) as usize;
        let chunk_type = &glb[cursor + 4..cursor + 8];
        let data_start = cursor + 8;
        let data_end = data_start + chunk_len;
        if data_end > glb.len() {
            return Err("chunk length runs past end of file".to_string());
        }
        let data = &glb[data_start..data_end];
        if chunk_type == b"JSON" {
            json_bytes = Some(data);
        } else if chunk_type == b"BIN\0" {
            bin_bytes = Some(data);
        }
        cursor = data_end;
    }

    let json_bytes = json_bytes.ok_or("missing JSON chunk")?;
    let bin_bytes = bin_bytes.ok_or("missing BIN chunk")?;

    let root: GltfRoot = serde_json::from_slice(json_bytes).map_err(|e| format!("bad glTF JSON: {e}"))?;

    let mesh = root.meshes.first().ok_or("no meshes in file")?;
    let primitive = mesh.primitives.first().ok_or("mesh has no primitives")?;

    let read_vec_accessor = |accessor_index: u32, components: usize| -> Result<Vec<f32>, String> {
        let accessor = root
            .accessors
            .get(accessor_index as usize)
            .ok_or_else(|| format!("accessor {accessor_index} out of range"))?;
        let view = root
            .buffer_views
            .get(accessor.buffer_view as usize)
            .ok_or("bufferView out of range")?;
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

    let indices_accessor = root
        .accessors
        .get(primitive.indices as usize)
        .ok_or("indices accessor out of range")?;
    let indices_view = root
        .buffer_views
        .get(indices_accessor.buffer_view as usize)
        .ok_or("indices bufferView out of range")?;
    let mut indices = Vec::with_capacity(indices_accessor.count);
    for i in 0..indices_accessor.count {
        indices.push(read_u32(bin_bytes, indices_view.byte_offset + i * 4));
    }

    let positions = flat_positions.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let normals = flat_normals.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let colors = flat_colors.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();

    Ok(ImportedMesh { positions, normals, colors, indices })
}