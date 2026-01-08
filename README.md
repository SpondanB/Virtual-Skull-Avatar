# Head Pose–Driven 3D Skull

## Overview
This project demonstrates a **real-time, camera-driven 3D animation system** where a virtual skull mirrors a user's **head orientation, eye blinks, and mouth movements** using a standard webcam.

It combines **computer vision**, **3D geometry**, and **real-time rendering**, showcasing practical skills in:
- Human–computer interaction
- Face tracking and pose estimation
- Lightweight 3D graphics without game engines

This project was built entirely from scratch using **Python**, without relying on high-level 3D engines such as Unity or Blender.

This project is an updated version of the previous [Face Tracker Project](https://github.com/SpondanB/FaceTracker) that I did. The final result of that project is added in the "Old version" directory.

---

## Key Features
- 🎯 **Real-time head pose tracking** (yaw, pitch, roll)
- 👁️ **Blink detection** with smooth eyelid animation
- 👄 **Mouth opening detection** driving jaw articulation
- 🦴 **Custom 3D skull mesh** rendered using pure math
- 💡 **Dynamic lighting and face-based shading**
- ⚡ Runs at ~60 FPS on a standard laptop webcam

---

## Technical Highlights
- **MediaPipe Face Mesh** for facial landmark detection
- **Perspective-n-Point (PnP)** for 3D head pose estimation
- **Manual 3D projection pipeline** (no OpenGL / no engines)
- **Z-buffer–like face sorting** for correct rendering order
- **Temporal smoothing** to prevent jitter and flipping
- **Procedural animation** for jaw and eyelids

---

## Skills Demonstrated
- Computer Vision (OpenCV, MediaPipe)
- Linear Algebra & 3D Transformations
- Real-Time Systems
- Graphics Math (normals, lighting, projection)
- Signal smoothing & noise handling
- Python performance optimization

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
|   ├── OldVer.py # Result of the previous attempt
├── Tests         # Test applications
|   ├── test-3d-obj.py
|   ├── test-combined-with-blinking.py
|   ├── test-combined.py
|   ├── test-rpy-calc.py
|   ├── test-thread-combined.py
├── Main.py        # Core application
├── README.md      # Project documentation
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

## Portfolio Use
This project is suitable for showcasing:
- AI / ML Master's applications
- Computer vision internships
- Graphics programming roles
- Human–computer interaction research

It demonstrates the ability to **bridge perception and graphics**, turning raw sensor data into expressive, real-time visual behavior.

---

## Author
**Spondan Bandyopadhyay**  
Interests: Computer Vision, Graphics, AI Systems

---
