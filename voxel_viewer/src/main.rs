use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

use voxel_core::{greedy_mesh, load_chunk, raycast_chunk, save_chunk, Chunk, MeshData, Voxel, CHUNK_SIZE};

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

/// Mouse-controlled orbit camera: left-drag rotates around `target`,
/// scroll zooms in/out. This is the standard "look at a 3D model" camera
/// you'd want for an asset editor, as opposed to a first-person flycam.
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

    /// dx/dy are raw cursor-movement deltas in pixels while dragging.
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
            target: Vec3::new(2.0, 1.5, 2.0),
            yaw: 45f32.to_radians(),
            pitch: 30f32.to_radians(),
            distance: 10.0,
        }
    }
}

/// Which effect a left-click has, matching the mockup's "Paint Mode" radio
/// group. Right-click stays a quick-remove shortcut regardless of mode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PaintMode {
    Add,
    Replace,
    Remove,
}

/// A single entry in the artist's palette: an id (what gets stored per
/// voxel), a display name, and an RGB swatch color.
#[derive(Clone)]
struct Material {
    id: u16,
    name: &'static str,
    color: [f32; 3],
}

/// Starting palette, matching the mockup's default 8 materials.
fn default_materials() -> Vec<Material> {
    vec![
        Material { id: 1, name: "White", color: [0.93, 0.93, 0.94] },
        Material { id: 2, name: "Coral", color: [0.92, 0.38, 0.38] },
        Material { id: 3, name: "Sun", color: [0.95, 0.80, 0.25] },
        Material { id: 4, name: "Leaf", color: [0.35, 0.75, 0.40] },
        Material { id: 5, name: "Sky", color: [0.30, 0.55, 0.90] },
        Material { id: 6, name: "Grape", color: [0.58, 0.38, 0.85] },
        Material { id: 7, name: "Charcoal", color: [0.28, 0.28, 0.31] },
        Material { id: 8, name: "Rust", color: [0.75, 0.42, 0.22] },
    ]
}

fn material_color(materials: &[Material], id: u16) -> [f32; 3] {
    materials
        .iter()
        .find(|m| m.id == id)
        .map(|m| m.color)
        .unwrap_or([0.6, 0.6, 0.6])
}

/// Builds a tiny demo shape: a 4x4 "Leaf" platform, a "Rust" pillar,
/// and a "Coral" cap on top -- used only when there's no save file yet.
fn build_demo_chunk() -> Chunk {
    let mut chunk = Chunk::empty();
    for x in 0..4 {
        for z in 0..4 {
            chunk.set(x, 0, z, Voxel::new(4)); // Leaf
        }
    }
    for y in 1..4 {
        chunk.set(1, y, 1, Voxel::new(8)); // Rust
        chunk.set(2, y, 1, Voxel::new(8));
        chunk.set(1, y, 2, Voxel::new(8));
        chunk.set(2, y, 2, Voxel::new(8));
    }
    for x in 0..3 {
        for z in 0..3 {
            chunk.set(x, 4, z, Voxel::new(2)); // Coral
        }
    }
    chunk
}

fn mesh_to_vertices(mesh: &MeshData, materials: &[Material]) -> Vec<Vertex> {
    mesh.positions
        .iter()
        .zip(mesh.normals.iter())
        .zip(mesh.voxel_ids.iter())
        .map(|((p, n), id)| Vertex { position: *p, normal: *n, color: material_color(materials, *id) })
        .collect()
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
}

fn create_depth_view(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

impl Gpu {
    /// v1 keeps this dead simple: one fixed save file next to the
    /// executable. Multi-file project management (New/Open/Save As) comes
    /// once there's more than a single chunk worth saving.
    const SAVE_PATH: &'static str = "voxel_save.bin";

    fn save(&self) {
        match save_chunk(&self.chunk, std::path::Path::new(Self::SAVE_PATH)) {
            Ok(()) => println!("saved to {}", Self::SAVE_PATH),
            Err(e) => eprintln!("save failed: {e}"),
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
                cull_mode: Some(wgpu::Face::Back),
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

        let chunk = match load_chunk(std::path::Path::new(Gpu::SAVE_PATH)) {
            Ok(c) => {
                println!("loaded existing save from {}", Gpu::SAVE_PATH);
                c
            }
            Err(e) => {
                println!("no usable save file ({e}) -- starting from the demo shape");
                build_demo_chunk()
            }
        };
        let materials = default_materials();
        let mesh = greedy_mesh(&chunk);
        let vertices = mesh_to_vertices(&mesh, &materials);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        println!(
            "loaded demo mesh: {} quads, {} verts, {} tris",
            mesh.quad_count(),
            mesh.positions.len(),
            mesh.triangle_count()
        );

        let egui_ctx = egui::Context::default();
        let egui_state = EguiState::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
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
            num_indices: mesh.indices.len() as u32,
            uniform_buffer,
            bind_group,
            window,
            camera: Camera::default(),
            chunk,
            materials,
            current_material: 5, // "Sky", matching the mockup's default selection
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
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
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

    /// Small movement threshold (in pixels) below which a press+release is
    /// treated as a "click" (place a voxel) rather than a camera drag.
    const CLICK_MOVE_THRESHOLD: f64 = 4.0;

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        match button {
            MouseButton::Left => {
                if state == ElementState::Pressed {
                    self.dragging = true;
                    self.drag_last = Some(self.cursor_pos);
                    self.press_pos = Some(self.cursor_pos);
                } else {
                    self.dragging = false;
                    self.drag_last = None;
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
        self.current_material = id;
        println!("current material: {id}");
    }

    /// Unprojects a screen-space cursor position into a world-space ray,
    /// using the inverse of the exact view-proj matrix the camera renders
    /// with -- so the ray always matches what's on screen, regardless of
    /// window size or camera position.
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

    /// Left-click entry point: does whatever the current PaintMode says to
    /// do at the clicked face -- add, replace-in-place, or remove.
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
        if current == new_voxel {
            return; // no-op edit -- don't pollute undo history
        }

        self.push_undo_snapshot();
        self.chunk.set(pos.x as usize, pos.y as usize, pos.z as usize, new_voxel);
        self.rebuild_mesh();
    }

    /// Right-click quick-remove, independent of whatever PaintMode is
    /// currently selected -- lets you erase without switching modes.
    fn try_remove_voxel(&mut self, x: f64, y: f64) {
        let (origin, dir) = self.cursor_ray(x, y);
        let Some(hit) = raycast_chunk(&self.chunk, origin, dir) else { return };
        self.push_undo_snapshot();
        self.chunk
            .set(hit.voxel.x as usize, hit.voxel.y as usize, hit.voxel.z as usize, Voxel::EMPTY);
        self.rebuild_mesh();
    }

    /// How many past states to keep. 50 undos is plenty for a single-chunk
    /// editor and costs almost nothing -- each snapshot is a Vec<u16> of
    /// 4096 voxel ids, ~8KB, so the whole stack tops out around 400KB.
    const MAX_HISTORY: usize = 50;

    /// Records the chunk's current state onto the undo stack before a
    /// mutation, and clears the redo stack.
    fn push_undo_snapshot(&mut self) {
        self.undo_stack.push(self.chunk.to_ids());
        if self.undo_stack.len() > Self::MAX_HISTORY {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    fn undo(&mut self) {
        let Some(prev_ids) = self.undo_stack.pop() else {
            println!("nothing to undo");
            return;
        };
        self.redo_stack.push(self.chunk.to_ids());
        if let Some(restored) = Chunk::from_ids(&prev_ids) {
            self.chunk = restored;
            self.rebuild_mesh();
        }
    }

    fn redo(&mut self) {
        let Some(next_ids) = self.redo_stack.pop() else {
            println!("nothing to redo");
            return;
        };
        self.undo_stack.push(self.chunk.to_ids());
        if let Some(restored) = Chunk::from_ids(&next_ids) {
            self.chunk = restored;
            self.rebuild_mesh();
        }
    }

    /// Re-runs the greedy mesher and re-uploads the vertex/index buffers.
    fn rebuild_mesh(&mut self) {
        let mesh = greedy_mesh(&self.chunk);
        let vertices = mesh_to_vertices(&mesh, &self.materials);

        self.vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index_buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.num_indices = mesh.indices.len() as u32;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.update_uniforms();

        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.08, g: 0.09, b: 0.11, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // --- egui overlay: paint mode + materials panel ---
        let raw_input = self.egui_state.take_egui_input(&*self.window);
        let materials = self.materials.clone();
        let mut selected = self.current_material;
        let mut paint_mode = self.paint_mode;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::SidePanel::right("materials_panel")
                .resizable(false)
                .default_width(220.0)
                .show(ctx, |ui| {
                    ui.heading("Paint Mode");
                    ui.separator();
                    ui.radio_value(&mut paint_mode, PaintMode::Add, "Add Voxels");
                    ui.radio_value(&mut paint_mode, PaintMode::Replace, "Replace Material");
                    ui.radio_value(&mut paint_mode, PaintMode::Remove, "Remove Voxels");

                    ui.add_space(12.0);
                    ui.heading("Materials");
                    ui.separator();
                    for m in &materials {
                        let is_selected = m.id == selected;
                        let swatch = egui::Color32::from_rgb(
                            (m.color[0] * 255.0) as u8,
                            (m.color[1] * 255.0) as u8,
                            (m.color[2] * 255.0) as u8,
                        );
                        ui.horizontal(|ui| {
                            let (rect, swatch_response) =
                                ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::click());
                            ui.painter().rect_filled(rect, 3.0, swatch);
                            let label_response =
                                ui.selectable_label(is_selected, format!("{}  #{:02}", m.name, m.id));
                            if swatch_response.clicked() || label_response.clicked() {
                                selected = m.id;
                            }
                        });
                    }
                    ui.separator();
                    let name = materials.iter().find(|m| m.id == selected).map(|m| m.name).unwrap_or("?");
                    ui.label(format!("Selected material: {name}"));
                    ui.add_space(12.0);
                    ui.label("Ctrl+S save · Ctrl+Z undo · Ctrl+Shift+Z redo");
                });
        });
        self.current_material = selected;
        self.paint_mode = paint_mode;

        self.egui_state.handle_platform_output(&*self.window, full_output.platform_output);
        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };
        self.egui_renderer
            .update_buffers(&self.device, &self.queue, &mut encoder, &clipped_primitives, &screen_descriptor);

        {
            let egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut egui_pass = egui_pass.forget_lifetime();
            self.egui_renderer.render(&mut egui_pass, &clipped_primitives, &screen_descriptor);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

#[derive(Default)]
struct App {
    gpu: Option<Gpu>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let window_attrs = Window::default_attributes().with_title("Voxel Engine - viewer v1");
        let window = Arc::new(event_loop.create_window(window_attrs).expect("create window"));
        self.gpu = Some(Gpu::new(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = &mut self.gpu else { return };

        let response = gpu.egui_state.on_window_event(&*gpu.window, &event);
        if response.repaint {
            gpu.window.request_redraw();
        }
        let consumed_by_ui = response.consumed;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::MouseInput { button, state, .. } => {
                if !consumed_by_ui {
                    gpu.handle_mouse_button(button, state);
                }
            }
            WindowEvent::CursorMoved { position, .. } => gpu.handle_cursor_moved(position.x, position.y),
            WindowEvent::MouseWheel { delta, .. } => {
                if !consumed_by_ui {
                    gpu.handle_scroll(delta);
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => gpu.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                if !consumed_by_ui && event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        match code {
                            KeyCode::Digit1 => gpu.set_material(1),
                            KeyCode::Digit2 => gpu.set_material(2),
                            KeyCode::Digit3 => gpu.set_material(3),
                            KeyCode::KeyS if gpu.modifiers.control_key() => gpu.save(),
                            KeyCode::KeyZ if gpu.modifiers.control_key() && gpu.modifiers.shift_key() => gpu.redo(),
                            KeyCode::KeyZ if gpu.modifiers.control_key() => gpu.undo(),
                            KeyCode::KeyY if gpu.modifiers.control_key() => gpu.redo(),
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                match gpu.render() {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        gpu.resize(gpu.config.width, gpu.config.height)
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("render error: {e:?}"),
                }
                gpu.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("create event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run app");
}