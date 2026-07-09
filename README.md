# 🧊 Voxel Engine (Viewer v1)

A high-performance, lightweight 3D voxel engine editor built from scratch in Rust using modern explicit graphics primitives. It features a custom greedy meshing engine, an optimized hardware-accelerated rendering pipeline via WGPU, and an embedded UI layer for rapid asset design and palette prototyping.

## 🚀 Architecture & Technical Highlights

* **Modern Graphics Stack:** Fully custom rendering engine utilizing `wgpu` targeting modern explicit APIs (Vulkan/Metal/DX12), moving away from legacy state machines into modern command encoders and explicit pipeline state objects (PSOs).
* **Greedy Meshing:** Implements an optimized greedy meshing algorithm that culls hidden faces and merges adjacent co-planar coplanar faces across a dense `16x16x16` local chunk, drastically reducing vertex layout overhead and minimizing batch submission sizes.
* **Deterministic Raycasting:** Built-in exact 3D AABB voxel raycasting loop calculating immediate intersecting voxel faces for deterministic building, replacing, and erasing actions.
* **State Management:** Low-overhead undo/redo snapshot architecture tracking a compact state history rolling stack (~8KB per snapshot layer).
* **Binary Persistence:** Direct serialization and deserialization via `bincode` to native byte streams with explicit binary versioning to guarantee backward compatibility on struct layout modifications.

---

## 🛠️ Controls & Workflow

### Mouse Navigation & Actions

| Input | Action | Description |
| --- | --- | --- |
| **Left Click + Drag** | **Orbit Camera** | Standard 3D viewer camera trackball orbit around the chunk target center. |
| **Mouse Wheel** | **Zoom** | Zoom smoothly in and out with clamped bounding distances. |
| **Left Click** | **Apply Paint** | Places, updates, or erases a voxel based on your selected sidebar **Paint Mode**. |
| **Middle Click** | **Material Picker** | Samples the material ID of the voxel directly under the cursor and shifts your current palette to match. |
| **Right Click** | **Quick Erase** | Direct shortcut to remove a voxel instantly, bypassing your current paint mode. |

### Hotkeys

* `Ctrl + S`: Commit state to disk (`voxel_save.bin`).
* `Ctrl + Z`: Undo last voxel alteration.
* `Ctrl + Shift + Z` or `Ctrl + Y`: Redo last undone alteration.
* `1` / `2` / `3`: Quick-swap active paint palette to Materials `#01`, `#02`, or `#03`.

---

## 📦 Project Structure

```text
├── main.rs              # Application entry point, Winit window loop, and WGPU render state management
├── shaders/
│   └── shader.wgsl      # Custom WGSL vertex and fragment shaders processing world transforms and lighting
└── voxel_core/          # Core voxel manipulation & processing engine
    ├── chunk.rs         # Dense array allocation, indexing arithmetic, and state tracking
    ├── voxel.rs         # Voxel primitive definitions and structural states
    ├── mesher.rs        # Greedy mesher face-merging implementation
    ├── raycast.rs       # 3D DDA/AABB intersection calculations for face targeting
    └── persistence.rs   # Binary I/O codec pipelines

```

---

## ⚡ Quick Start

Ensure you have the latest stable [Rust toolchain installed](https://rustup.rs/).

### Development Build

```bash
cargo check

```

### Run Client Viewer

Always run the engine with optimization flags enabled. The greedy meshing architecture relies on optimized compiler loops to construct meshes instantly when large edits occur.

```bash
cargo run --release

```
