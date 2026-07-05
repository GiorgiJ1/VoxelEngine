use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use voxel_core::{greedy_mesh, Chunk, MeshData, Voxel};

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
        // Clamp pitch just short of straight up/down so the view never
        // flips upside down when you drag past the pole.
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
            target: Vec3::new(2.0, 1.5, 2.0), // roughly the center of the demo shape
            yaw: 45f32.to_radians(),
            pitch: 30f32.to_radians(),
            distance: 10.0,
        }
    }
}

/// Placeholder palette: voxel id -> RGB. This will eventually be a real,
/// artist-editable palette; for now it's just enough to prove multiple
/// materials render distinctly.
fn palette(id: u16) -> [f32; 3] {
    match id {
        1 => [0.30, 0.75, 0.35], // green "grass" base
        2 => [0.80, 0.45, 0.20], // orange "wood" pillar
        3 => [0.85, 0.15, 0.15], // red "roof" cap
        _ => [0.6, 0.6, 0.6],
    }
}

/// Builds a tiny demo shape: a 4x4 green platform, a single orange pillar,
/// and a red cap on top -- enough to show greedy-merged flat faces AND
/// multiple materials in one mesh.
fn build_demo_chunk() -> Chunk {
    let mut chunk = Chunk::empty();
    for x in 0..4 {
        for z in 0..4 {
            chunk.set(x, 0, z, Voxel::new(1));
        }
    }
    for y in 1..4 {
        chunk.set(1, y, 1, Voxel::new(2));
        chunk.set(2, y, 1, Voxel::new(2));
        chunk.set(1, y, 2, Voxel::new(2));
        chunk.set(2, y, 2, Voxel::new(2));
    }
    for x in 0..3 {
        for z in 0..3 {
            chunk.set(x, 4, z, Voxel::new(3));
        }
    }
    chunk
}

fn mesh_to_vertices(mesh: &MeshData) -> Vec<Vertex> {
    mesh.positions
        .iter()
        .zip(mesh.normals.iter())
        .zip(mesh.voxel_ids.iter())
        .map(|((p, n), id)| Vertex { position: *p, normal: *n, color: palette(*id) })
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
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
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

        let chunk = build_demo_chunk();
        let mesh = greedy_mesh(&chunk);
        let vertices = mesh_to_vertices(&mesh);

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
            dragging: false,
            last_cursor: None,
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

    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.dragging = state == ElementState::Pressed;
            if !self.dragging {
                self.last_cursor = None;
            }
        }
    }

    fn handle_cursor_moved(&mut self, x: f64, y: f64) {
        if self.dragging {
            if let Some((last_x, last_y)) = self.last_cursor {
                let dx = (x - last_x) as f32;
                let dy = (y - last_y) as f32;
                self.camera.orbit(dx, dy);
            }
        }
        self.last_cursor = Some((x, y));
    }

    fn handle_scroll(&mut self, delta: MouseScrollDelta) {
        let amount = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => (pos.y / 100.0) as f32,
        };
        self.camera.zoom(amount);
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
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => gpu.resize(size.width, size.height),
            WindowEvent::MouseInput { button, state, .. } => gpu.handle_mouse_button(button, state),
            WindowEvent::CursorMoved { position, .. } => gpu.handle_cursor_moved(position.x, position.y),
            WindowEvent::MouseWheel { delta, .. } => gpu.handle_scroll(delta),
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