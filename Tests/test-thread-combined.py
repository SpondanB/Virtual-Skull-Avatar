import threading
import math
import numpy as np
import pygame
import cv2
import mediapipe as mp

# ============================================================
# Issues
# 1. the face tracking is not really moving the object correctly :(
# 2. need to fix the roll pitch yaw issue.
# ============================================================

# ============================================================
# Shared Head Pose State
# ============================================================
shared_angles = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
lock = threading.Lock()

# ============================================================
# OpenCV + MediaPipe Head Pose Thread
# ============================================================
def head_pose_thread():

    def rotationMatrixToEulerAngles(R):
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2,1], R[2,2])
            pitch = math.atan2(-R[2,0], sy)
            yaw = math.atan2(R[1,0], R[0,0])
        else:
            roll = math.atan2(-R[1,2], R[1,1])
            pitch = math.atan2(-R[2,0], sy)
            yaw = 0
        return np.array([roll, pitch, yaw])

    class AngleSmoother:
        def __init__(self, alpha=0.85):
            self.alpha = alpha
            self.prev = None
        def smooth(self, a):
            if self.prev is None:
                self.prev = a
                return a
            self.prev = self.alpha * self.prev + (1 - self.alpha) * a
            return self.prev

    model_points = np.array([
        (0,0,0),(0,-330,-65),
        (-225,170,-135),(225,170,-135),
        (-150,-150,-125),(150,-150,-125)
    ], dtype=np.float32)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    smoother = AngleSmoother(0.8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            image_points = np.array([
                (lm[1].x*w, lm[1].y*h),
                (lm[152].x*w, lm[152].y*h),
                (lm[33].x*w, lm[33].y*h),
                (lm[263].x*w, lm[263].y*h),
                (lm[61].x*w, lm[61].y*h),
                (lm[291].x*w, lm[291].y*h)
            ], dtype=np.float32)

            cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float32)
            success, rvec, _ = cv2.solvePnP(
                model_points, image_points, cam, np.zeros((4,1))
            )

            if success:
                R, _ = cv2.Rodrigues(rvec)
                angles = np.degrees(rotationMatrixToEulerAngles(R))
                angles = smoother.smooth(angles)

                with lock:
                    shared_angles["roll"]  = angles[0]
                    shared_angles["pitch"] = angles[1]
                    shared_angles["yaw"]   = angles[2]

        cv2.imshow("Head Pose (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# Start Head Pose Thread
# ============================================================
threading.Thread(target=head_pose_thread, daemon=True).start()

# ============================================================
# Pygame Renderer
# ============================================================
pygame.init()
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Head-Tracked Skull")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 14)

# ---------- Geometry ----------
vertices = np.array([
    (-0.65,  0.65,  0.35), ( 0.65,  0.65,  0.35),
    ( 0.55,  0.65, -0.45), (-0.55,  0.65, -0.45),
    (-0.85,  0.10,  0.55), ( 0.85,  0.10,  0.55),
    ( 0.75,  0.10, -0.65), (-0.75,  0.10, -0.65),
    (-0.55, -0.55,  0.35), ( 0.55, -0.55,  0.35),
    ( 0.45, -0.55, -0.45), (-0.45, -0.55, -0.45),

    (-0.38,  0.12,  0.62), (-0.18,  0.12,  0.62),
    (-0.18, -0.12,  0.62), (-0.38, -0.12,  0.62),

    ( 0.18,  0.12,  0.62), ( 0.38,  0.12,  0.62),
    ( 0.38, -0.12,  0.62), ( 0.18, -0.12,  0.62),

    (-0.45, -0.90,  0.30), ( 0.45, -0.90,  0.30),
    ( 0.30, -1.35,  0.10), (-0.30, -1.35,  0.10),

    ( 0.00, -0.05,  0.72), (-0.14, -0.28,  0.60),
    ( 0.14, -0.28,  0.60),
], dtype=np.float32)

vertices /= np.max(np.linalg.norm(vertices, axis=1))


faces = [
    (0,1,2),(0,2,3),
    (4,5,1),(4,1,0),
    (12,13,14),(12,14,15),
    (16,17,18),(16,18,19),
    (0,3,7),(0,7,4),
    (1,5,6),(1,6,2),
    (3,2,6),(3,6,7),
    (8,9,10),(8,10,11),
    (8,9,21),(8,21,20),
    (20,21,22),(20,22,23),
    (24,25,26),
    (13,24,25),
    (16,26,24),
    # sides
    (4, 7, 11), (4, 11, 8),
    (6, 5, 9), (6, 9, 10),
    # front
    (5, 4, 8), (5, 8, 9),
    # back
    (7, 6, 10), (7, 10, 11),
    # jaw side covers
    (10, 9, 21), (10, 21, 22),
    (8, 11, 20), (20, 11, 23),
    # jaw back cover
    (11, 10, 22), (11,22,23),
]

face_colors = [
    (235,235,235),(235,235,235),
    (225,225,225),(225,225,225),
    (30,30,30),(30,30,30),
    (30,30,30),(30,30,30),
    (210,210,210),(210,210,210),
    (210,210,210),(210,210,210),
    (195,195,195),(195,195,195),
    (185,185,185),(185,185,185),
    (175,175,175),(175,175,175),
    (165,165,165),(165,165,165),
    (40,40,40),
    (205,205,205),
    (205,205,205),
    # new colors
    (210,210,210),(210,210,210),
    (210,210,210),(210,210,210),
    (210,210,210),(210,210,210),
    (210,210,210),(210,210,210),
    # jaw colors
    (205,205,205),
    (205,205,205),
    (205,205,205),
    (205,205,205),
    (175,175,175),(175,175,175),
]

# ---------- Math ----------
def rotation_matrix(yaw, pitch, roll):
    Rx = np.array([[1,0,0],
                   [0,math.cos(pitch),-math.sin(pitch)],
                   [0,math.sin(pitch), math.cos(pitch)]])
    Ry = np.array([[ math.cos(yaw),0,math.sin(yaw)],
                   [0,1,0],
                   [-math.sin(yaw),0,math.cos(yaw)]])
    Rz = np.array([[math.cos(roll),-math.sin(roll),0],
                   [math.sin(roll), math.cos(roll),0],
                   [0,0,1]])
    return Ry @ Rx @ Rz   # correct order

def project(v):
    fov = 500
    z = v[2] + 4
    return (int(v[0]*fov/z + WIDTH/2),
            int(-v[1]*fov/z + HEIGHT/2)), z

def face_normal(a,b,c):
    return np.cross(b-a, c-a)

light_dir = np.array([0.3,0.5,-1])
light_dir /= np.linalg.norm(light_dir)
ambient = 0.5

show_indices = False
running = True

# ============================================================
# Main Loop
# ============================================================
while running:
    clock.tick(60)
    screen.fill((20,20,20))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_i:
            show_indices = not show_indices

    with lock:
        raw_yaw   = shared_angles["yaw"]
        raw_pitch = shared_angles["pitch"]
        raw_roll  = shared_angles["roll"]

    # ---- Corrected mapping ----
    yaw   = math.radians(-raw_yaw)   * 0.8
    pitch = math.radians(-raw_pitch) * 0.8
    roll  = math.radians(raw_roll)   * 0.5

    pitch = np.clip(pitch, -0.9, 0.9)
    yaw   = np.clip(yaw,   -1.2, 1.2)

    # ---- Face camera correction ----
    face_camera = rotation_matrix(math.pi, 0, 0)
    R = face_camera @ rotation_matrix(yaw, pitch, roll)

    rotated = [R @ v for v in vertices]
    projected = [project(v) for v in rotated]

    face_data = []
    for i, face in enumerate(faces):
        pts3d = [rotated[j] for j in face]
        pts2d = [projected[j][0] for j in face]
        avg_z = sum(projected[j][1] for j in face) / 3
        n = face_normal(*pts3d)
        n /= np.linalg.norm(n)
        intensity = ambient + (1-ambient)*max(0, np.dot(n, -light_dir))
        color = np.clip(np.array(face_colors[i]) * intensity, 0, 255)
        face_data.append((avg_z, pts2d, color))

    face_data.sort(reverse=True, key=lambda x: x[0])
    for _, pts, color in face_data:
        pygame.draw.polygon(screen, color, pts)

    if show_indices:
        for i, ((x,y),_) in enumerate(projected):
            screen.blit(font.render(str(i), True, (255,0,0)), (x+4,y+4))

    pygame.display.flip()

pygame.quit()
