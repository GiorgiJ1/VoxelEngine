struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = uniforms.model * vec4<f32>(model.position, 1.0);
    out.world_pos = world_pos.xyz;
    out.normal = (uniforms.model * vec4<f32>(model.normal, 0.0)).xyz;
    out.color = model.color;
    out.clip_position = uniforms.view_proj * world_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Basic Directional Lighting
    let light_dir = normalize(vec3<f32>(0.4, 1.0, 0.3));
    let ambient = 0.45;
    let diff = max(dot(normalize(in.normal), light_dir), 0.0) * 0.55;
    let lighting = ambient + diff;
    
    var final_color = in.color * lighting;

    // Procedural Floor Grid (Evaluated near y = 0 surface)
    if (in.world_pos.y <= 0.005 && in.world_pos.y >= -0.005) {
        let coord = in.world_pos.xz;
        let grid_derivative = fwidth(coord);
        let grid_lines = abs(fract(coord - 0.5) - 0.5) / grid_derivative;
        let line_factor = min(grid_lines.x, grid_lines.y);
        
        // Compute anti-aliased line blending
        let line_intensity = 1.0 - min(line_factor, 1.0);
        
        // Primary axes colors (X = Red, Z = Blue), secondary lines = Dark Slate
        var grid_color = vec3<f32>(0.22, 0.23, 0.26);
        if (abs(in.world_pos.z) < 0.05) {
            grid_color = vec3<f32>(0.75, 0.25, 0.25); // X Axis Line
        } else if (abs(in.world_pos.x) < 0.05) {
            grid_color = vec3<f32>(0.25, 0.45, 0.75); // Z Axis Line
        }

        final_color = mix(final_color, grid_color, line_intensity * 0.65);
    }

    return vec4<f32>(final_color, 1.0);
}