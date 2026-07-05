struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.model * vec4<f32>(in.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    out.world_normal = (uniforms.model * vec4<f32>(in.normal, 0.0)).xyz;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 0.9, 0.4));
    let n = normalize(in.world_normal);
    let diffuse = max(dot(n, light_dir), 0.0);
    let ambient = 0.35;
    let lit = in.color * (ambient + diffuse * 0.65);
    return vec4<f32>(lit, 1.0);
}