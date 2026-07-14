# Voxel Engine Sandbox

A modern voxel editor and rendering engine built from scratch in **Rust**, powered by **wgpu**, **winit**, and **egui**.

The project demonstrates how to build a complete voxel pipeline without relying on a game engine—from voxel storage and greedy meshing to GPU rendering, raycasting, editing tools, and 3D model export.

---

# Preview

> <img width="1917" height="1017" alt="image" src="https://github.com/user-attachments/assets/1520b4da-f985-4bf9-8c87-c2e7c724ccf3" />



![Editor](docs/editor.png)

---

# Features

### Rendering

- GPU accelerated rendering using **wgpu**
- Modern **WebGPU** rendering pipeline
- Real-time lighting
- Orbit camera controls
- Infinite viewport grid
- High-performance mesh rebuilding

### Voxel Editing

- Add voxels
- Paint existing voxels
- Remove voxels
- 255 material palette
- Brush size controls
- Mirror painting (X / Y / Z)
- Undo / Redo system
- Real-time mesh updates

### Engine

- 32×32×32 voxel chunk
- Greedy Meshing optimization
- Fast voxel raycasting
- Material system
- Binary save/load format
- Efficient GPU mesh generation

### Import / Export

- Save project (`.bin`)
- Load project (`.bin`)
- Export glTF (`.glb`)
- Export Wavefront (`.obj + .mtl`)

---

# Project Structure

```
VoxelEngine/
│
├── voxel_core/
│   ├── chunk.rs
│   ├── mesher.rs
│   ├── raycast.rs
│   ├── export.rs
│   └── lib.rs
│
└── voxel_viewer/
    ├── shaders/
    │   └── shader.wgsl
    └── main.rs
```

### voxel_core

Contains the engine itself.

Responsible for:

- voxel storage
- greedy meshing
- raycasting
- exporting
- save/load
- shared engine types

### voxel_viewer

Desktop editor built using:

- wgpu
- winit
- egui

Responsible for:

- rendering
- editor UI
- camera
- input
- GPU resources
- editor tools

---

# Controls

## Camera

| Input | Action |
|--------|--------|
| Middle Mouse Drag | Orbit camera |
| Mouse Wheel | Zoom |

## Editing

| Input | Action |
|--------|--------|
| Left Click | Paint |
| Right Click | Remove voxel |

## Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl + S | Save |
| Ctrl + Z | Undo |
| Ctrl + Y | Redo |
| Ctrl + E | Export GLB |
| 1–8 | Select material |

---

# Tech Stack

- Rust
- wgpu
- winit
- egui
- glam
- bytemuck
- pollster
- rfd

---

# Performance

The engine is designed around efficient rendering techniques.

- Greedy Meshing drastically reduces polygon count.
- Meshes are rebuilt only when voxel data changes.
- GPU vertex/index buffers are regenerated on demand.
- Rendering uses indexed geometry for minimal bandwidth.

---

# Roadmap

## Planned

- [ ] Infinite chunk streaming
- [ ] Multiple chunk support
- [ ] Ambient Occlusion
- [ ] Texture atlas support
- [ ] PBR materials
- [ ] Scene lighting editor
- [ ] Gizmos
- [ ] Selection tool
- [ ] Copy / Paste
- [ ] Fill tool
- [ ] Noise terrain generator
- [ ] Animation support
- [ ] Plugin system

---

# Building

Clone the repository

```bash
git clone https://github.com/yourname/VoxelEngine.git
cd VoxelEngine
```

Build

```bash
cargo build --workspace
```

Run

```bash
cargo run -p voxel_viewer
```

---

# Why This Project?

This project was built to explore low-level graphics programming in Rust and understand how professional voxel engines work internally.

Instead of relying on existing engines, every major system—from mesh generation to rendering and editor tooling—was implemented from scratch.

---

# License

MIT License
