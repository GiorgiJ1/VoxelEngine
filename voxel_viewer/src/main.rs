use std::sync::Arc;
use std::path::PathBuf;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use voxel_core::{
    export_gltf_glb, export_obj_mtl, greedy_mesh, load_chunk, raycast_chunk, save_chunk, Chunk, MeshData, Voxel, CHUNK_SIZE,
};

use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use egui_winit::State as EguiState;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x3 },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

struct Camera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl Camera {
    fn eye(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }

    fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.1, 100.0);
        proj * view
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.005;
        self.yaw -= dx * SENSITIVITY;
        self.pitch = (self.pitch + dy * SENSITIVITY).clamp(-1.5, 1.5);
    }

    fn zoom(&mut self, scroll_amount: f32) {
        const ZOOM_SPEED: f32 = 0.5;
        self.distance = (self.distance - scroll_amount * ZOOM_SPEED).clamp(2.0, 40.0);
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::new((CHUNK_SIZE / 2) as f32, 0.0, (CHUNK_SIZE / 2) as f32),
            yaw: 45f32.to_radians(),
            pitch: 30f32.to_radians(),
            distance: 24.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PaintMode {
    Add,
    Replace,
    Remove,
}

#[derive(Clone)]
struct Material {
    id: u16,
    name: &'static str,
    color: [f32; 3],
    opacity_pct: f32,
    metallic_pct: f32,
}

// Helper function to generate full 255 color palette
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match (h * 6.0) as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}

fn default_materials() -> Vec<Material> {
    let mut mats = Vec::new();
    
    // Core explicitly targeted colors for the pedastal
    mats.push(Material { id: 1, name: "Red", color: [0.8, 0.25, 0.25], opacity_pct: 100.0, metallic_pct: 0.0 });
    mats.push(Material { id: 2, name: "Green", color: [0.35, 0.65, 0.25], opacity_pct: 100.0, metallic_pct: 0.0 });
    mats.push(Material { id: 3, name: "Blue", color: [0.25, 0.45, 0.85], opacity_pct: 100.0, metallic_pct: 0.0 });
    mats.push(Material { id: 4, name: "White", color: [0.95, 0.95, 0.95], opacity_pct: 100.0, metallic_pct: 0.0 });

    // Procedurally generated grid of standard MagicVoxel-style colors
    let mut id = 5;
    
    // Grayscale
    for i in 0..12 {
        let v = i as f32 / 11.0;
        mats.push(Material { id: id as u16, name: "Gray", color: [v, v, v], opacity_pct: 100.0, metallic_pct: 0.0 });
        id += 1;
    }

    // Color swatches
    for h in 0..12 {
        for s in 0..4 {
            for l in 0..5 {
                if id > 255 { break; }
                let hue = h as f32 / 12.0;
                let sat = 1.0 - (s as f32 / 4.0);
                let light = 0.2 + (l as f32 / 5.0) * 0.7;
                let (r, g, b) = hsl_to_rgb(hue, sat, light);
                mats.push(Material { id: id as u16, name: "Color", color: [r, g, b], opacity_pct: 100.0, metallic_pct: 0.0 });
                id += 1;
            }
        }
    }
    
    // Fill remaining to ensure exactly 255 materials available in palette
    while id <= 255 {
        mats.push(Material { id: id as u16, name: "Color", color: [0.5, 0.5, 0.5], opacity_pct: 100.0, metallic_pct: 0.0 });
        id += 1;
    }
    
    mats
}

fn material_color(materials: &[Material], id: u16) -> [f32; 3] {
    materials
        .iter()
        .find(|m| m.id == id)
        .map(|m| m.color)
        .unwrap_or([0.6, 0.6, 0.6])
}

fn build_demo_chunk() -> Chunk {
    let mut chunk = Chunk::empty();
    
    let center_x = CHUNK_SIZE / 2;
    let center_z = CHUNK_SIZE / 2;
    
    // Base platform (Green)
    let base_radius = 6;
    for x in (center_x - base_radius)..=(center_x + base_radius) {
        for z in (center_z - base_radius)..=(center_z + base_radius) {
            chunk.set(x, 0, z, Voxel::new(2)); 
        }
    }
    
    // Pedestal Bottom Layer (Red)
    for x in (center_x - 2)..=(center_x + 2) {
        for z in (center_z - 2)..=(center_z + 2) {
            chunk.set(x, 1, z, Voxel::new(1));
        }
    }
    
    // Pillar Column (Red)
    for y in 2..7 {
        for x in (center_x - 1)..=(center_x + 1) {
            for z in (center_z - 1)..=(center_z + 1) {
                chunk.set(x, y, z, Voxel::new(1));
            }
        }
    }
    
    // Pedestal Top Layer (Red)
    for x in (center_x - 2)..=(center_x + 2) {
        for z in (center_z - 2)..=(center_z + 2) {
            chunk.set(x, 7, z, Voxel::new(1));
        }
    }
    
    chunk
}

fn append_grid_geometry(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>) {
    let start_idx = vertices.len() as u32;
    let extent = 64.0; // Defines spatial domain boundary for grid projection layout
    
    // Low-overhead quad positioned at ground plane (y=0) to receive the procedural shader grid
    vertices.push(Vertex { position: [-extent, 0.0, -extent], normal: [0.0, 1.0, 0.0], color: [0.14, 0.15, 0.17] });
    vertices.push(Vertex { position: [ extent, 0.0, -extent], normal: [0.0, 1.0, 0.0], color: [0.14, 0.15, 0.17] });
    vertices.push(Vertex { position: [ extent, 0.0,  extent], normal: [0.0, 1.0, 0.0], color: [0.14, 0.15, 0.17] });
    vertices.push(Vertex { position: [-extent, 0.0,  extent], normal: [0.0, 1.0, 0.0], color: [0.14, 0.15, 0.17] });

    indices.extend_from_slice(&[
        start_idx, start_idx + 2, start_idx + 1,
        start_idx, start_idx + 3, start_idx + 2,
    ]);
}

fn mesh_to_vertices(mesh: &MeshData, materials: &[Material]) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices: Vec<Vertex> = mesh.positions
        .iter()
        .zip(mesh.normals.iter())
        .zip(mesh.voxel_ids.iter())
        .map(|((p, n), id)| Vertex { position: *p, normal: *n, color: material_color(materials, *id) })
        .collect();

    let mut indices = mesh.indices.clone();
    append_grid_geometry(&mut vertices, &mut indices);
    (vertices, indices)
}

struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    depth_view: wgpu::TextureView,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    window: Arc<Window>,
    camera: Camera,
    chunk: Chunk,
    materials: Vec<Material>,
    current_material: u16,
    paint_mode: PaintMode,
    dragging: bool,
    cursor_pos: (f64, f64),
    drag_last: Option<(f64, f64)>,
    press_pos: Option<(f64, f64)>,
    modifiers: ModifiersState,
    undo_stack: Vec<Vec<u16>>,
    redo_stack: Vec<Vec<u16>>,
    egui_ctx: egui::Context,
    egui_state: EguiState,
    egui_renderer: EguiRenderer,
    current_project_path: Option<PathBuf>,
    brush_size: i32,
    mirror_x: bool,
    mirror_y: bool,
    mirror_z: bool,
    last_frame_instant: Instant,
    fps: f32,
    last_mesh_build_ms: f32,
}

impl Gpu {
    const SAVE_PATH: &'static str = "voxel_save.bin";
    const EXPORT_PATH: &'static str = "voxel_export.glb";
    const OBJ_EXPORT_BASE: &'static str = "voxel_export";

    fn save(&self) {
        match save_chunk(&self.chunk, std::path::Path::new(Self::SAVE_PATH)) {
            Ok(()) => println!("saved to {}", Self::SAVE_PATH),
            Err(e) => eprintln!("save failed: {e}"),
        }
    }

    fn build_material_resolver(&self) -> [[f32; 3]; 256] {
        let mut resolver = [[0.5f32; 3]; 256];
        for m in &self.materials {
            if (m.id as usize) < 256 {
                resolver[m.id as usize] = m.color;
            }
        }
        resolver
    }

    fn export(&self) {
        let mesh = greedy_mesh(&self.chunk);
        let resolver = self.build_material_resolver();
        match export_gltf_glb(&mesh, std::path::Path::new(Self::EXPORT_PATH), &resolver) {
            Ok(()) => println!("Exported the glTF to {}", Self::EXPORT_PATH),
            Err(e) => eprintln!("export of the glTF failed: {e}"),
        }
    }
    
    fn export_obj(&self) {
        let mesh = greedy_mesh(&self.chunk);
        let resolver = self.build_material_resolver();
        match export_obj_mtl(&mesh, std::path::Path::new(Self::OBJ_EXPORT_BASE), &resolver) {
            Ok(()) => println!("exported OBJ/MTL group to {}.obj/.mtl", Self::OBJ_EXPORT_BASE),
            Err(e) => eprintln!("export OBJ failed: {e}"),
        }
    }

    fn file_new(&mut self) {
        self.chunk = Chunk::empty();
        self.current_project_path = None;
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.rebuild_mesh();
    }

    fn file_open(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Voxel Project File (*.bin)", &["bin"])
            .pick_file()
        {
            if let Ok(loaded_chunk) = load_chunk(&path) {
                self.chunk = loaded_chunk;
                self.current_project_path = Some(path);
                self.undo_stack.clear();
                self.redo_stack.clear();
                self.rebuild_mesh();
            }
        }
    }

    fn file_save(&mut self) {
        if self.current_project_path.is_some() {
            let path = self.current_project_path.clone().unwrap();
            self.perform_save_to_path(&path);
        } else {
            self.file_save_as();
        }
    }

    fn file_save_as(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("project")
            .add_filter("Voxel Project File (*.bin)", &["bin"])
            .add_filter("glTF 2.0 Binary Container (*.glb)", &["glb"])
            .add_filter("Wavefront Structural Layout (*.obj)", &["obj"])
            .save_file()
        {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                match ext.to_lowercase().as_str() {
                    "bin" => {
                        self.perform_save_to_path(&path);
                        self.current_project_path = Some(path);
                    }
                    "glb" => {
                        let mesh = greedy_mesh(&self.chunk);
                        let resolver = self.build_material_resolver();
                        let _ = export_gltf_glb(&mesh, &path, &resolver);
                    }
                    "obj" => {
                        let mesh = greedy_mesh(&self.chunk);
                        let resolver = self.build_material_resolver();
                        let _ = export_obj_mtl(&mesh, &path, &resolver);
                    }
                    _ => {}
                }
            }
        }
    }

    fn perform_save_to_path(&self, path: &std::path::Path) {
        let _ = save_chunk(&self.chunk, path);
    }

    fn file_export_glb(&self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("model_export.glb")
            .add_filter("glTF 2.0 Binary Container (*.glb)", &["glb"])
            .save_file()
        {
            let mesh = greedy_mesh(&self.chunk);
            let resolver = self.build_material_resolver();
            let _ = export_gltf_glb(&mesh, &path, &resolver);
        }
    }

    fn file_export_obj(&self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("model_export")
            .add_filter("Wavefront Structural Layout Bundle (*.obj)", &["obj"])
            .save_file()
        {
            let mesh = greedy_mesh(&self.chunk);
            let resolver = self.build_material_resolver();
            let _ = export_obj_mtl(&mesh, &path, &resolver);
        }
    }

    fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no suitable GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_view = create_depth_view(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let chunk = build_demo_chunk();
        let materials = default_materials();
        let initial_build_start = Instant::now();
        let mesh = greedy_mesh(&chunk);
        let (vertices, indices) = mesh_to_vertices(&mesh, &materials);
        let initial_mesh_build_ms = initial_build_start.elapsed().as_secs_f32() * 1000.0;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let egui_ctx = egui::Context::default();
        
        let mut visuals = egui::Visuals::dark();
        // Background palette matching modern DCC viewports (Blender/MagicaVoxel layout structures)
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(20, 21, 23);
        visuals.widgets.noninteractive.weak_bg_fill = egui::Color32::from_rgb(24, 25, 28);
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(40, 42, 45));
        
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(32, 34, 38);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(45, 48, 54);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(0, 120, 255); // Highlight structural interactions blue

        visuals.window_fill = egui::Color32::from_rgb(18, 19, 21);
        visuals.panel_fill = egui::Color32::from_rgb(18, 19, 21);
        egui_ctx.set_visuals(visuals);

        let egui_state = EguiState::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = EguiRenderer::new(&device, config.format, None, 1, false);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            depth_view,
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            uniform_buffer,
            bind_group,
            window,
            camera: Camera::default(),
            chunk,
            materials,
            current_material: 1,
            paint_mode: PaintMode::Add,
            dragging: false,
            cursor_pos: (0.0, 0.0),
            drag_last: None,
            press_pos: None,
            modifiers: ModifiersState::default(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            egui_ctx,
            egui_state,
            egui_renderer,
            current_project_path: None,
            brush_size: 1,
            mirror_x: false,
            mirror_y: false,
            mirror_z: false,
            last_frame_instant: Instant::now(),
            fps: 0.0,
            last_mesh_build_ms: initial_mesh_build_ms,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, &self.config);
    }

    fn update_uniforms(&self) {
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let view_proj = self.camera.view_proj(aspect);
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    const CLICK_MOVE_THRESHOLD: f64 = 4.0;

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        match button {
            MouseButton::Middle => {
                if state == ElementState::Pressed {
                    self.dragging = true;
                    self.drag_last = Some(self.cursor_pos);
                } else {
                    self.dragging = false;
                    self.drag_last = None;
                }
            }
            MouseButton::Left => {
                if state == ElementState::Pressed {
                    self.press_pos = Some(self.cursor_pos);
                } else if state == ElementState::Released {
                    if let Some(press) = self.press_pos.take() {
                        let moved = ((self.cursor_pos.0 - press.0).powi(2)
                            + (self.cursor_pos.1 - press.1).powi(2))
                        .sqrt();
                        if moved < Self::CLICK_MOVE_THRESHOLD {
                            self.apply_paint(self.cursor_pos.0, self.cursor_pos.1);
                        }
                    }
                }
            }
            MouseButton::Right => {
                if state == ElementState::Released {
                    self.try_remove_voxel(self.cursor_pos.0, self.cursor_pos.1);
                }
            }
            _ => {}
        }
    }

    fn handle_cursor_moved(&mut self, x: f64, y: f64) {
        self.cursor_pos = (x, y);
        if self.dragging {
            if let Some((last_x, last_y)) = self.drag_last {
                let dx = (x - last_x) as f32;
                let dy = (y - last_y) as f32;
                self.camera.orbit(dx, dy);
            }
            self.drag_last = Some((x, y));
        }
    }

    fn handle_scroll(&mut self, delta: MouseScrollDelta) {
        let amount = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => (pos.y / 100.0) as f32,
        };
        self.camera.zoom(amount);
    }

    fn set_material(&mut self, id: u16) {
        if id > 0 && id <= 255 {
            self.current_material = id;
        }
    }

    fn cursor_ray(&self, x: f64, y: f64) -> (Vec3, Vec3) {
        let width = self.config.width as f32;
        let height = self.config.height.max(1) as f32;
        let ndc_x = (2.0 * x as f32 / width) - 1.0;
        let ndc_y = 1.0 - (2.0 * y as f32 / height);
        let aspect = width / height;
        let inv_vp = self.camera.view_proj(aspect).inverse();
        let near = inv_vp * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near_world = near.truncate() / near.w;
        let far_world = far.truncate() / far.w;
        (near_world, (far_world - near_world).normalize())
    }

    fn apply_paint(&mut self, x: f64, y: f64) {
        let (origin, dir) = self.cursor_ray(x, y);
        let Some(hit) = raycast_chunk(&self.chunk, origin, dir) else { return };

        let target: Option<(IVec3, Voxel)> = match self.paint_mode {
            PaintMode::Add => hit.place_at.and_then(|p| {
                let size = CHUNK_SIZE as i32;
                if p.x < 0 || p.y < 0 || p.z < 0 || p.x >= size || p.y >= size || p.z >= size {
                    None
                } else {
                    Some((p, Voxel::new(self.current_material)))
                }
            }),
            PaintMode::Replace => Some((hit.voxel, Voxel::new(self.current_material))),
            PaintMode::Remove => Some((hit.voxel, Voxel::EMPTY)),
        };

        let Some((pos, new_voxel)) = target else { return };
        let current = self.chunk.get(pos.x as usize, pos.y as usize, pos.z as usize);
        if current == new_voxel { return; }

        self.push_undo_snapshot();
        self.chunk.set(pos.x as usize, pos.y as usize, pos.z as usize, new_voxel);
        self.rebuild_mesh();
    }

    fn try_remove_voxel(&mut self, x: f64, y: f64) {
        let (origin, dir) = self.cursor_ray(x, y);
        let Some(hit) = raycast_chunk(&self.chunk, origin, dir) else { return };
        self.push_undo_snapshot();
        self.chunk.set(hit.voxel.x as usize, hit.voxel.y as usize, hit.voxel.z as usize, Voxel::EMPTY);
        self.rebuild_mesh();
    }

    const MAX_HISTORY: usize = 50;

    fn push_undo_snapshot(&mut self) {
        self.undo_stack.push(self.chunk.to_ids());
        if self.undo_stack.len() > Self::MAX_HISTORY {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    fn undo(&mut self) {
        let Some(prev_ids) = self.undo_stack.pop() else { return; };
        self.redo_stack.push(self.chunk.to_ids());
        if let Some(restored) = Chunk::from_ids(&prev_ids) {
            self.chunk = restored;
            self.rebuild_mesh();
        }
    }

    fn redo(&mut self) {
        let Some(next_ids) = self.redo_stack.pop() else { return; };
        self.undo_stack.push(self.chunk.to_ids());
        if let Some(restored) = Chunk::from_ids(&next_ids) {
            self.chunk = restored;
            self.rebuild_mesh();
        }
    }

    fn rebuild_mesh(&mut self) {
        let build_start = Instant::now();
        let mesh = greedy_mesh(&self.chunk);
        let (vertices, indices) = mesh_to_vertices(&mesh, &self.materials);

        self.vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.num_indices = indices.len() as u32;
        self.last_mesh_build_ms = build_start.elapsed().as_secs_f32() * 1000.0;
    }

    fn clear_chunk(&mut self) {
        self.push_undo_snapshot();
        self.chunk = Chunk::empty();
        self.rebuild_mesh();
    }

    fn voxel_count(&self) -> usize {
        let mut count = 0;
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    if !self.chunk.get(x, y, z).is_empty() {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
            let now = Instant::now();
            let dt = now.duration_since(self.last_frame_instant).as_secs_f32();
            self.last_frame_instant = now;
            if dt > 0.0 {
                let instant_fps = 1.0 / dt;
                self.fps = if self.fps == 0.0 { instant_fps } else { self.fps * 0.9 + instant_fps * 0.1 };
            }

            self.update_uniforms();

            let frame = self.surface.get_current_texture()?;
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.1,
                                b: 0.1,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if self.num_indices > 0 {
                    rpass.set_pipeline(&self.pipeline);
                    rpass.set_bind_group(0, &self.bind_group, &[]);
                    rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                    rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    rpass.draw_indexed(0..self.num_indices, 0, 0..1);
                }
            }

            let egui_input = self.egui_state.take_egui_input(&self.window);
            self.egui_ctx.begin_pass(egui_input);

            let ctx = self.egui_ctx.clone();

            let mut paint_mode = self.paint_mode;
            let mut selected = self.current_material;
            let materials_edit = self.materials.clone();
            let mut brush_size = self.brush_size;
            let mut mirror_x = self.mirror_x;
            let mut mirror_y = self.mirror_y;
            let mut mirror_z = self.mirror_z;

            // --- Top Menu Bar ---
            egui::TopBottomPanel::top("top_panel").show(&&ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("New").clicked() {
                            self.file_new();
                            ui.close_menu();
                        }
                        if ui.button("Open...").clicked() {
                            self.file_open();
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Save").clicked() {
                            self.file_save();
                            ui.close_menu();
                        }
                        if ui.button("Save As...").clicked() {
                            self.file_save_as();
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Export GLB...").clicked() {
                            self.file_export_glb();
                            ui.close_menu();
                        }
                        if ui.button("Export OBJ...").clicked() {
                            self.file_export_obj();
                            ui.close_menu();
                        }
                    });

                    ui.menu_button("Edit", |ui| {
                        if ui.button("Undo").clicked() {
                            self.undo();
                            ui.close_menu();
                        }
                        if ui.button("Redo").clicked() {
                            self.redo();
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Clear Voxel Canvas").clicked() {
                            self.clear_chunk();
                            ui.close_menu();
                        }
                    });
                });
            });

            // Modernizing panel framing properties
            let panel_frame = egui::Frame {
                fill: ctx.style().visuals.panel_fill,
                inner_margin: egui::Margin::same(14.0),
                stroke: egui::Stroke::new(1.0, egui::Color32::from_rgb(35, 37, 40)),
                ..Default::default()
            };
            
            // --- Left Toolbar (Tools) ---
            egui::SidePanel::left("left_panel")
                .frame(panel_frame)
                .exact_width(180.0)
                .resizable(false)
                .show(&ctx, |ui| {
                    ui.label(egui::RichText::new("BRUSH").strong().color(egui::Color32::WHITE));
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.selectable_label(paint_mode == PaintMode::Add, "Attach").clicked() { paint_mode = PaintMode::Add; }
                        if ui.selectable_label(paint_mode == PaintMode::Replace, "Paint").clicked() { paint_mode = PaintMode::Replace; }
                        if ui.selectable_label(paint_mode == PaintMode::Remove, "Erase").clicked() { paint_mode = PaintMode::Remove; }
                    });
                    
                    ui.add_space(10.0);
                    ui.add(egui::Slider::new(&mut brush_size, 1..=10).text("Size"));
                    ui.add_space(20.0);

                    ui.label(egui::RichText::new("MIRROR").strong().color(egui::Color32::WHITE));
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.toggle_value(&mut mirror_x, "X");
                        ui.toggle_value(&mut mirror_y, "Y");
                        ui.toggle_value(&mut mirror_z, "Z");
                    });
            });

            // --- Right Toolbar (Palette & Materials) ---
            egui::SidePanel::right("right_panel")
                .frame(panel_frame)
                .exact_width(260.0)
                .resizable(false)
                .show(&ctx, |ui| {
                    // 1. ACTIVE MATERIAL PREVIEW
                    ui.label(egui::RichText::new("ACTIVE MATERIAL").strong().color(egui::Color32::WHITE));
                    ui.add_space(4.0);
                    
                    if let Some(active_mat) = self.materials.iter().find(|m| m.id == selected) {
                        ui.horizontal(|ui| {
                            // Big color swatch preview
                            let (r, g, b) = ((active_mat.color[0] * 255.0) as u8, (active_mat.color[1] * 255.0) as u8, (active_mat.color[2] * 255.0) as u8);
                            let preview_color = egui::Color32::from_rgb(r, g, b);
                            let (rect, _) = ui.allocate_exact_size(egui::vec2(40.0, 40.0), egui::Sense::hover());
                            ui.painter().rect_filled(rect, 6.0, preview_color);
                            ui.painter().rect_stroke(rect, 6.0, egui::Stroke::new(1.0, egui::Color32::from_white_alpha(40)));

                            ui.vertical(|ui| {
                                ui.label(egui::RichText::new(active_mat.name).strong().color(egui::Color32::WHITE));
                                ui.label(egui::RichText::new(format!("ID: #{:03}", active_mat.id)).weak());
                            });
                        });
                    }
                    
                    ui.add_space(16.0);

                    // 2. PALETTE TABS
                    ui.label(egui::RichText::new("PALETTE").strong().color(egui::Color32::WHITE));
                    ui.separator();
                    
                    ui.horizontal(|ui| {
                        let _ = ui.selectable_label(true, "Project");
                        let _ = ui.selectable_label(false, "Custom");
                        let _ = ui.selectable_label(false, "History");
                    });
                    ui.add_space(6.0);
                    
                    // 3. SWATCH GRID (Scrollable)
                    let available_height = ui.available_height() - 140.0; // Reserve space for properties below
                    egui::ScrollArea::vertical().max_height(available_height).show(ui, |ui| {
                        let swatch_size = egui::vec2(21.0, 21.0);
                        ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                        
                        ui.horizontal_wrapped(|ui| {
                            for mat in &materials_edit {
                                let color = egui::Color32::from_rgb(
                                    (mat.color[0] * 255.0) as u8,
                                    (mat.color[1] * 255.0) as u8,
                                    (mat.color[2] * 255.0) as u8,
                                );
                                
                                let is_selected = selected == mat.id;
                                let response = ui.allocate_response(swatch_size, egui::Sense::click());
                                if response.clicked() {
                                    selected = mat.id;
                                }
                                
                                let mut rect = response.rect;
                                
                                // Hover Animation: Expand slightly
                                if response.hovered() {
                                    rect = rect.expand(1.5);
                                }
                                
                                // Render Swatch Base
                                ui.painter().rect_filled(rect, 4.0, color);
                                
                                // Selection Accent / Border Styling
                                if is_selected {
                                    // Modern crisp neon blue selection border
                                    let blue_accent = egui::Color32::from_rgb(0, 140, 255);
                                    ui.painter().rect_stroke(rect, 4.0, egui::Stroke::new(1.0, blue_accent));
                                    ui.painter().rect_stroke(rect.expand(1.5), 5.0, egui::Stroke::new(1.0, egui::Color32::WHITE));
                                } else if response.hovered() {
                                    ui.painter().rect_stroke(rect, 4.0, egui::Stroke::new(1.5, egui::Color32::WHITE));
                                } else {
                                    ui.painter().rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::from_black_alpha(80)));
                                }
                            }
                        });
                    });

                    ui.add_space(16.0);

                    // 4. MATERIAL PROPERTIES SLIDERS
                    ui.label(egui::RichText::new("MATERIAL PROPERTIES").strong().color(egui::Color32::WHITE));
                    ui.separator();
                    
                    if let Some(idx) = self.materials.iter().position(|m| m.id == selected) {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Opacity").weak());
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.add(egui::Slider::new(&mut self.materials[idx].opacity_pct, 0.0..=100.0).text("%").show_value(true));
                            });
                        });
                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Metallic").weak());
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.add(egui::Slider::new(&mut self.materials[idx].metallic_pct, 0.0..=100.0).text("%").show_value(true));
                            });
                        });
                    }
            });

            // --- Bottom Infobar ---
            let bottom_frame = egui::Frame::side_top_panel(&&ctx.style()).inner_margin(egui::Margin::symmetric(16.0, 6.0));
            egui::TopBottomPanel::bottom("bottom_panel").frame(bottom_frame).show(&ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Dimensions: {}x{}x{}", CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE));
                    ui.separator();
                    ui.label(format!("Voxels: {}", self.voxel_count()));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!("{:.1} FPS", self.fps));
                        ui.separator();
                        ui.label(format!("Build ms: {:.2}", self.last_mesh_build_ms));
                        ui.separator();
                        ui.label(if self.undo_stack.is_empty() { "Saved" } else { "Unsaved changes" });
                    });
                });
            });

            self.paint_mode = paint_mode;
            if selected != self.current_material {
                self.set_material(selected);
            }
            self.brush_size = brush_size;
            self.mirror_x = mirror_x;
            self.mirror_y = mirror_y;
            self.mirror_z = mirror_z;

            let output = self.egui_ctx.end_pass();
            let paint_jobs = self.egui_ctx.tessellate(output.shapes, self.egui_ctx.pixels_per_point());

            let screen_descriptor = ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            };

            for (id, image_delta) in &output.textures_delta.set {
                self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
            }

            self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &paint_jobs, &screen_descriptor);

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui render pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                let mut rpass = rpass.forget_lifetime();

                self.egui_renderer
                    .render(&mut rpass, &paint_jobs, &screen_descriptor);
            }

            self.queue.submit(Some(encoder.finish()));
            frame.present();

            for id in &output.textures_delta.free {
                self.egui_renderer.free_texture(id);
            }

            Ok(())
    }
}

fn create_depth_view(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let desc = wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let texture = device.create_texture(&desc);
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

struct App {
    gpu: Option<Gpu>,
}

impl App {
    fn new() -> Self {
        Self { gpu: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let attrs = Window::default_attributes().with_title("Voxel Editor");
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.gpu = Some(Gpu::new(window));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let gpu = match self.gpu.as_mut() {
            Some(g) if g.window.id() == id => g,
            _ => return,
        };

        let response = gpu.egui_state.on_window_event(&gpu.window, &event);
        if response.consumed {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                gpu.resize(physical_size.width, physical_size.height);
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                gpu.modifiers = modifiers.state();
            }
            WindowEvent::KeyboardInput { event: kb_event, .. } => {
                if kb_event.state == ElementState::Pressed {
                    match kb_event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyS) if gpu.modifiers.control_key() => gpu.file_save(),
                        PhysicalKey::Code(KeyCode::KeyZ) if gpu.modifiers.control_key() => gpu.undo(),
                        PhysicalKey::Code(KeyCode::KeyY) if gpu.modifiers.control_key() => gpu.redo(),
                        PhysicalKey::Code(KeyCode::KeyE) if gpu.modifiers.control_key() => gpu.file_export_glb(),
                        PhysicalKey::Code(KeyCode::Digit1) => gpu.set_material(1),
                        PhysicalKey::Code(KeyCode::Digit2) => gpu.set_material(2),
                        PhysicalKey::Code(KeyCode::Digit3) => gpu.set_material(3),
                        PhysicalKey::Code(KeyCode::Digit4) => gpu.set_material(4),
                        PhysicalKey::Code(KeyCode::Digit5) => gpu.set_material(5),
                        PhysicalKey::Code(KeyCode::Digit6) => gpu.set_material(6),
                        PhysicalKey::Code(KeyCode::Digit7) => gpu.set_material(7),
                        PhysicalKey::Code(KeyCode::Digit8) => gpu.set_material(8),
                        _ => {}
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                gpu.handle_cursor_moved(position.x, position.y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                gpu.handle_mouse_button(button, state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                gpu.handle_scroll(delta);
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = gpu.render() {
                    eprintln!("Render error: {e}");
                }
                gpu.window.request_redraw();
            }
            _ => {}
        }
    }
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    let _ = event_loop.run_app(&mut app);
}