import pygame
import numpy as np
import math

# ---------- Setup ----------
pygame.init()
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Skull model")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 16)

# ---------- Geometry ----------
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
    # ( 0.45, -0.55, -0.45), (-0.45, -0.55, -0.45), # 10, 11 repeat for jaw // 27, 28
    # (-0.45, -0.90,  0.30), ( 0.45, -0.90,  0.30), # 20, 21 repeat for jaw // 29, 30
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
    (20,21,22),(20,22,23), # older version
    # (29,30,22),(29,22,23),
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
    (10, 9, 21), (10, 21, 22), # older version
    (8, 11, 20), (20, 11, 23), # older version
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

# ---------- Rotation ----------
yaw = pitch = roll = 0.0
speed = 0.03

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
    return Rz @ Ry @ Rx

def project(v):
    fov = 500
    z = v[2] + 4
    x = v[0] * fov / z + WIDTH / 2
    y = -v[1] * fov / z + HEIGHT / 2
    return (int(x), int(y)), z

def face_normal(a,b,c):
    return np.cross(b-a, c-a)

# ---------- Lighting ----------
light_dir = np.array([0.3, 0.5, -1])
light_dir /= np.linalg.norm(light_dir)
ambient = 0.5

# ---------- Toggle ----------
show_vertex_indices = False

# ---------- Main Loop ----------
running = True
while running:
    clock.tick(60)
    screen.fill((20,20,20))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_i:
                show_vertex_indices = not show_vertex_indices

    keys = pygame.key.get_pressed()
    if keys[pygame.K_q]: pitch += speed
    if keys[pygame.K_e]: pitch -= speed
    if keys[pygame.K_a]: yaw += speed
    if keys[pygame.K_d]: yaw -= speed
    if keys[pygame.K_z]: roll += speed
    if keys[pygame.K_c]: roll -= speed

    R = rotation_matrix(yaw, pitch, roll)
    rotated = [R @ v for v in vertices]
    projected = [project(v) for v in rotated]

    # ----- Draw faces -----
    face_data = []
    for i, face in enumerate(faces):
        pts_3d = [rotated[idx] for idx in face]
        pts_2d = [projected[idx][0] for idx in face]
        avg_z = sum(projected[idx][1] for idx in face) / 3

        n = face_normal(*pts_3d)
        n /= np.linalg.norm(n)
        intensity = max(0, np.dot(n, -light_dir))
        intensity = ambient + (1 - ambient) * intensity
        color = np.clip(np.array(face_colors[i]) * intensity, 0, 255)

        face_data.append((avg_z, pts_2d, color))

    face_data.sort(reverse=True, key=lambda x: x[0])

    for _, pts, color in face_data:
        pygame.draw.polygon(screen, color, pts)

    # ----- Draw vertex indices -----
    if show_vertex_indices:
        for i, ((x, y), z) in enumerate(projected):
            label = font.render(str(i), True, (255, 0, 0))
            screen.blit(label, (x + 4, y + 4))

    pygame.display.flip()

pygame.quit()
