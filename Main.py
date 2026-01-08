import cv2
import mediapipe as mp
import numpy as np
import pygame
import math

# ============================================================
# MediaPipe Head Pose Setup
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

def rotation_matrix_to_angles(R):
    pitch = math.atan2(R[2,1], R[2,2])
    yaw   = math.atan2(-R[2,0], math.sqrt(R[0,0]**2 + R[1,0]**2))
    roll  = math.atan2(R[1,0], R[0,0])
    return np.array([pitch, yaw, roll])

# 3D reference face points (arbitrary but consistent)
model_points = np.array([
    [0, 0, 0],        # Nose
    [0, -330, -65],   # Chin
    [-225, 170, -135],# Left eye
    [225, 170, -135], # Right eye
    [-150, -150, -125],
    [150, -150, -125]
], dtype=np.float64)

# ============================================================
# Pygame Setup
# ============================================================
pygame.init()
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Head Pose Driven Skull")
clock = pygame.time.Clock()

# ============================================================
# Skull Geometry
# ============================================================
vertices = np.array([
    (-0.65,  0.65,  0.35), ( 0.65,  0.65,  0.35), # 0, 1
    ( 0.55,  0.65, -0.45), (-0.55,  0.65, -0.45), # 2, 3
    (-0.85,  0.10,  0.55), ( 0.85,  0.10,  0.55), # 4, 5
    ( 0.75,  0.10, -0.65), (-0.75,  0.10, -0.65), # 6, 7
    (-0.55, -0.55,  0.35), ( 0.55, -0.55,  0.35), # 8, 9
    ( 0.45, -0.55, -0.45), (-0.45, -0.55, -0.45), # 10, 11

    (-0.38,  0.12,  0.62), (-0.18,  0.12,  0.62), # 12, 13
    (-0.18, -0.12,  0.62), (-0.38, -0.12,  0.62), # 14, 15

    ( 0.18,  0.12,  0.62), ( 0.38,  0.12,  0.62), # 16, 17
    ( 0.38, -0.12,  0.62), ( 0.18, -0.12,  0.62), # 18, 19

    (-0.45, -0.90,  0.30), ( 0.45, -0.90,  0.30), # 20, 21
    ( 0.30, -1.35,  0.10), (-0.30, -1.35,  0.10), # 22, 23

    ( 0.00, -0.05,  0.72), (-0.14, -0.28,  0.60), # 24, 25
    ( 0.14, -0.28,  0.60),                        # 26
    ( 0.45, -0.55, -0.45), (-0.45, -0.55, -0.45), # 10, 11 repeat for jaw // 27, 28
    (-0.45, -0.90,  0.30), ( 0.45, -0.90,  0.30), # 20, 21 repeat for jaw // 29, 30
], dtype=np.float32)

vertices /= np.max(np.linalg.norm(vertices, axis=1))

faces = [
    (0,1,2),(0,2,3),
    (4,5,1),(4,1,0),
    (12,13,14),(12,14,15), # left eye
    (16,17,18),(16,18,19), # right eye
    (0,3,7),(0,7,4),
    (1,5,6),(1,6,2),
    (3,2,6),(3,6,7),
    (8,9,10),(8,10,11),
    (8,9,21),(8,21,20),
    # (20,21,22),(20,22,23), # older version
    (29,30,22),(29,22,23),
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
    (10, 9, 21), (27, 30, 22), # (10, 21, 22), # older version
    (8, 11, 20), (29, 28, 23), # (20, 11, 23), # older version
    # jaw back cover
    (28, 27, 22), (28,22,23),
    # mouth upper plate
    (20, 21, 11), (21, 10, 11), 
    # mount bottom plate
    (29, 30, 28), (30, 27, 28),
]

face_colors = [
    (235,235,235),(235,235,235),
    (225,225,225),(225,225,225),
    # eyes
    (30,30,30),(30,30,30),
    (30,30,30),(30,30,30),

    (210,210,210),(210,210,210),
    (210,210,210),(210,210,210),
    (210,210,210),(210,210,210), # (195,195,195),(195,195,195),
    (210,210,210),(210,210,210), # (185,185,185),(185,185,185),
    (205,205,205),(205,205,205), # (175,175,175),(175,175,175),
    (205,205,205),(205,205,205), # (165,165,165),(165,165,165),
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
    # mouth colors
    (168, 50, 74),(168, 50, 74),
    (168, 50, 74),(168, 50, 74),
]

# for eye blink thingy
LEFT_EYE_FACES  = [4, 5]
RIGHT_EYE_FACES = [6, 7]
EYE_FACES = LEFT_EYE_FACES + RIGHT_EYE_FACES
# ey open / close representation
EYE_OPEN_COLOR   = np.array([40, 40, 40])   # dark socket
EYE_CLOSED_COLOR = np.array([200, 200, 200])  # dimmed / eyelid-like

# Fixed correction to align skull with camera coordinates
CORRECTION_ROT = np.array([
    [1,  0,  0],
    [0, -1,  0],   # flip Y axis
    [0,  0, 1]    
])

prev_R = None


JAW_VERTICES = [
    29, 30, 22, 23,   # lower jaw front
    27, 28            # jaw sides
]


# ============================================================
# Math Utilities
# ============================================================
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
    return Ry @ Rx @ Rz   # correct human-head order

def project(v):
    fov = 500
    z = v[2] + 4
    return (int(v[0]*fov/z + WIDTH/2),
            int(-v[1]*fov/z + HEIGHT/2)), z

def face_normal(a,b,c):
    return np.cross(b-a, c-a)

# Lighting
light_dir = np.array([0.3,0.5,-1])
light_dir /= np.linalg.norm(light_dir)
ambient = 0.5

# blink utilities
def eye_aspect_ratio(lm, upper, lower, left, right, w, h):
    up = np.array([lm[upper].x * w, lm[upper].y * h])
    lo = np.array([lm[lower].x * w, lm[lower].y * h])
    le = np.array([lm[left].x  * w, lm[left].y  * h])
    ri = np.array([lm[right].x * w, lm[right].y * h])

    vert = np.linalg.norm(up - lo)
    horiz = np.linalg.norm(le - ri)
    return vert / horiz

BLINK_THRESH = 0.23
blink_l = blink_r = False
blink_strength = 0.0   # 0 = open, 1 = fully closed

# mouth opening utilities
def mouth_open_ratio(lm, upper, lower, w, h):
    up = np.array([lm[upper].x * w, lm[upper].y * h])
    lo = np.array([lm[lower].x * w, lm[lower].y * h])
    return np.linalg.norm(up - lo)

jaw_open = 0.0          # smoothed jaw openness (0–1)
JAW_OPEN_SCALE = 0.235  # translation strength (tune this)

# ============================================================
# Main Loop
# ============================================================
running = True
pitch , yaw , roll = 0.0, 0.0, 0.0

while running:
    clock.tick(60)
    screen.fill((20,20,20))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ---------- Head Pose ----------
    success, image = cap.read()
    if not success:
        continue

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # blinking stuff
        ear_left = eye_aspect_ratio(lm, 159, 145, 33, 133, w, h)
        ear_right = eye_aspect_ratio(lm, 386, 374, 362, 263, w, h)

        blink_l = ear_left < BLINK_THRESH
        blink_r = ear_right < BLINK_THRESH
        blink = blink_l or blink_r

        # Smooth blink value (avoids flicker)
        target = 1.0 if blink else 0.0
        blink_strength += (target - blink_strength) * 0.4
        # =============================================================================
        # -------- Mouth openness --------
        mouth_dist = mouth_open_ratio(lm, 13, 14, w, h)

        # normalize (these values work well for most webcams)
        mouth_norm = np.clip((mouth_dist - 5) / 20.0, 0.0, 1.0)

        # smooth motion (critical for realism)
        jaw_open += (mouth_norm - jaw_open) * 0.35
        # =============================================================================
        
        image_points = np.array([
            (lm[1].x*w, lm[1].y*h),
            (lm[152].x*w, lm[152].y*h),
            (lm[33].x*w, lm[33].y*h),
            (lm[263].x*w, lm[263].y*h),
            (lm[61].x*w, lm[61].y*h),
            (lm[291].x*w, lm[291].y*h)
        ], dtype=np.float64)

        cam_matrix = np.array([[w,0,w/2],
                               [0,w,h/2],
                               [0,0,1]], dtype=np.float64)

        _, rvec, _ = cv2.solvePnP(
            model_points, image_points, cam_matrix, np.zeros((4,1))
        )
        R, _ = cv2.Rodrigues(rvec)

        # solution to fix the random flipping ======================================
        if prev_R is not None:
            delta = np.linalg.norm(R - prev_R)
            if delta > 1.5:   # threshold
                R = prev_R
        else:
            prev_R = R

        prev_R = R.copy()
        # ==========================================================================
        angles = rotation_matrix_to_angles(R)

        angles = rotation_matrix_to_angles(R)

        pitch = -angles[0]
        yaw   = -angles[1]
        roll  = angles[2]
        # print(f"pitch: {pitch}; yaw: {yaw}; roll {roll}") # is working fine as expected

    # ---------- Apply Rotation ----------
    R_head = rotation_matrix(yaw, pitch, roll)
    R = R_head @ CORRECTION_ROT
    # rotated = [R @ v for v in vertices] # regular version
    # updated to show jaw motion
    rotated = []
    for idx, v in enumerate(vertices):
        v_rot = R @ v

        # ---------- JAW TRANSLATION ----------
        if idx in JAW_VERTICES:
            v_rot = v_rot + np.array([
                0.0,                       # X: no sideways motion
                -jaw_open * JAW_OPEN_SCALE,# Y: downwards
                jaw_open * JAW_OPEN_SCALE * 0.4  # Z: slight forward
            ])

        rotated.append(v_rot)
    # ==========================================================

    projected = [project(v) for v in rotated]

    face_data = []
    for i, face in enumerate(faces):
        pts3d = [rotated[j] for j in face]
        pts2d = [projected[j][0] for j in face]
        avg_z = sum(projected[j][1] for j in face) / 3
        n = face_normal(*pts3d)
        n /= np.linalg.norm(n)
        intensity = ambient + (1-ambient)*max(0, np.dot(n, -light_dir))

        # color = np.clip(np.array(face_colors[i]) * intensity, 0, 255) # regular version
        # updated to show blinking
        base_color = np.array(face_colors[i])

        # ----- Blink eye dimming -----
        if i in EYE_FACES:
            base_color = (
                (1 - blink_strength) * EYE_OPEN_COLOR +
                blink_strength * EYE_CLOSED_COLOR
            )

        color = np.clip(base_color * intensity, 0, 255)
        face_data.append((avg_z, pts2d, color))

    face_data.sort(reverse=True, key=lambda x: x[0])
    for _, pts, color in face_data:
        pygame.draw.polygon(screen, color, pts)

    pygame.display.flip()

cap.release()
pygame.quit()
