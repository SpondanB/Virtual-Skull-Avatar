# Head Pose–Driven 3D Skull

## Overview
This project demonstrates a **real-time, camera-driven 3D animation system** where a virtual skull mirrors a user's **head orientation, eye blinks, and mouth movements** using a standard webcam.

It combines **computer vision**, **3D geometry**, and **real-time rendering**, showcasing practical skills in:
- Human–computer interaction
- Face tracking and pose estimation
- Lightweight 3D graphics (using pygame) without game engines
- Physics-inspired Particle Systems

This project was built entirely from scratch using **Python**, without relying on high-level 3D engines such as Unity or Blender.

This project is an updated version of the previous [Face Tracker Project](https://github.com/SpondanB/FaceTracker) that I did. The final result of that project is added in the "Old version" directory.

---

## Key Features

| Feature                | Description                                    |
| ---------------------- | ---------------------------------------------- |
| 🧭 Head Pose Tracking  | Pitch, Yaw, Roll control the skull orientation |
| 👁 Blink Detection     | Eye Aspect Ratio controls eyelid animation     |
| 👄 Mouth Tracking      | Jaw opens and closes with your mouth           |
| 💀 Custom 3D skull mesh      | Rendered using just the mathematical coordinates |
| ✨ Particle Aura        | Procedural floating particles under the jaw    |
| 💡 Lighting System     | Face normal lighting with depth sorting        |
| 🎮 Real-Time Rendering | 60 FPS Pygame 3D renderer                      |

---

## Technical Highlights
- **MediaPipe Face Mesh** for facial landmark detection
- **Perspective-n-Point (PnP)** for 3D head pose estimation
- **Manual 3D projection pipeline** (no OpenGL / no engines)
- **Z-buffer–like face sorting** for correct rendering order
- **Temporal smoothing** to prevent jitter and flipping
- **Procedural animation** for jaw and eyelids

---

## 🧩 System Architecture

```
Webcam → MediaPipe Face Mesh → SolvePnP → Rotation Matrix
                     ↓
         Blink + Mouth Detection
                     ↓
     3D Skull Transformation + Jaw Rig
                     ↓
     Lighting + Projection + Z-Sorting
                     ↓
           Pygame Renderer
```

---

## 🧠 Key Concepts Demonstrated

### 1. Head Pose Estimation

* 3D face model + 2D landmarks
* `cv2.solvePnP()` estimates camera-space rotation
* Rodrigues → Rotation Matrix → Euler Angles

### 2. 3D Rendering Pipeline

* Custom vertex buffer
* Face normal lighting
* Perspective projection
* Depth sorting (Painter’s Algorithm)

### 3. Facial Animation

* Eye Aspect Ratio → Blink animation
* Mouth distance → Jaw rigging
* Smooth interpolation for realism

### 4. Particle System

* Procedural particle emission
* Physics-based motion
* Lifetime fading
* Alpha blending

---

## Installation

### Requirements
- A standard device with Python (3.10) installed
- mentioned list of python packages
- Webcam

### Install dependencies
```bash
pip install requirements.txt
```

---

## How to Run
```bash
python Main.py
```

Controls:
- Move your head → skull rotates
- Blink → eye sockets dim
- Open mouth → jaw moves

Press **close** on the window to exit.

---

## Project Structure
```
├── Old version
|   └── OldVer.py # Result of the previous attempt
├── Tests         # Test applications
|   ├── test-3d-obj.py
|   ├── test-combined-with-blinking.py
|   ├── test-combined.py
|   ├── test-rpy-calc.py
|   └── test-thread-combined.py
├── Main.py        # Core application
└── README.md      # Project documentation
```

---

## Design Decisions
- **No external 3D engines**: to demonstrate raw understanding of 3D math
- **Low-level rendering**: polygons drawn manually via Pygame
- **Robust pose smoothing**: prevents sudden 180° flips common in PnP
- **Physically inspired motion**: jaw motion uses proportional translation

---

## Potential Extensions
- Texture-mapped faces
- Emotion-based facial deformation
- Audio-driven lip-sync
- Export to OpenGL / WebGL
- Multi-face tracking

---

## ⭐ Why This Project Matters

This project showcases:

* Advanced applied linear algebra
* Real-time computer vision pipelines
* Interactive 3D graphics from scratch

---

## Author
**Spondan Bandyopadhyay**  
Interests: Computer Vision, Graphics, AI Systems, Human-Computer Interaction

---
