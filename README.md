Here is a clean, comprehensive `README.md` file for your workspace root. It documents the architecture, the multi-material export capabilities (`.glb` and `.obj`/`.mtl`), user interface keybindings, and instructions on how to build and run the engine.

### `README.md`

Create a file named `README.md` in your project's root directory (`G:\Downloads\VoxelEngine-main\VoxelEngine-main\README.md`) and add the following content:

```markdown
# Retro Voxel Engine & Viewer

A lightweight, high-performance 3D voxel engine built from scratch in Rust using modern low-level systems libraries. The project utilizes `wgpu` for pure hardware-accelerated rendering via WebGPU standards and `winit` for low-overhead window handling, featuring a clean retro arcade aesthetic side panel rendered with `egui`.

The engine implements a multi-material architecture, custom asset saving/loading configurations, a fast 3D cross-section mesher, and versatile 3D asset exporters.

---

## 🏗️ Project Architecture

The engine is structured as a cargo workspace containing two decoupled crates:

```text
├── voxel_core/            # Core Game Engine Logic & Shared Data Layouts
│   ├── src/
│   │   ├── chunk.rs       # 32x32x32 uniform spatial subdivision buffers
│   │   ├── mesher.rs      # Greedy meshing optimization logic
│   │   ├── raycast.rs     # Ray-aabb intersection for voxel grid placement
│   │   ├── export.rs      # Multi-material GLB, OBJ, and MTL encoders
│   │   └── lib.rs         # Public workspace API surface definitions
│   └── Cargo.toml
│
└── voxel_viewer/          # Graphical Subsystem & Application Entry point
    ├── src/
    │   ├── shaders/       # WGSL hardware pipeline shader targets
    │   └── main.rs        # WGPU device bindings, state management, and egui loops
    └── Cargo.toml

```

---

## 🚀 Key Features

* **Greedy Meshing Optimization:** Reduces spatial face counts significantly by combining raw neighboring voxel cubes sharing matching material data properties into single optimized stretch quads.
* **Dual-Format Asset Exporters:**
* **Compact glTF 2.0 Binary (`.glb`):** Packaged self-contained binary outputs containing coordinate systems and inline layout accessors for explicit RGB vertex color pipelines.
* **Split Multi-Material Wavefront (`.obj`/`.mtl`):** Generates structural mesh face descriptions partitioned perfectly by material index references mapping directly onto standard ambient/diffuse material library definition logs.


* **Non-Destructive State Undo/Redo Engine:** Captures chronological 50-step historical layout buffers allowing instant backward and forward spatial tracking safely.
* **Orbit Camera System:** High-density raycasting pipelines supporting sub-voxel tracking for additive painting, material replacement, or precise face elimination.

---

## 🎮 Interface & Keybindings

Interact with the viewport using your mouse and quick hotkeys:

### Camera Control

* **Left-Click + Drag:** Orbit around look-at target coordinate.
* **Scroll Wheel:** Zoom camera focus inward or outward.

### Editor Operations

* **Left-Click:** Paint action (triggers Add, Replace, or Remove based on current selection in side-panel).
* **Right-Click:** Quick-Remove voxel shortcut (bypasses current tool mode).

### Quick Hotkeys

| Keybinding | Action |
| --- | --- |
| `1`, `2`, `3` | Instantly hot-swap to active material indices |
| `Ctrl + S` | Save current chunk state onto disk (`voxel_save.bin`) |
| `Ctrl + E` | Export active mesh to binary glTF layout (`voxel_export.glb`) |
| `Ctrl + Z` | Undo last voxel structural paint manipulation |
| `Ctrl + Shift + Z` / `Ctrl + Y` | Redo previously reverted paint change operation |

---

## 🛠️ Building and Running

Ensure you have the latest stable Rust toolchain setup on your system environment.

### Compile the Workspace

To build all binary assets and dependent engine library layers inside the workspace cleanly, run:

```bash
cargo build --workspace

```

### Execute the Engine Viewer

To spin up the WebGPU pipeline window wrapper and run the voxel editor instance directly, execute:

```bash
cargo run -p voxel_viewer

```

```

```
