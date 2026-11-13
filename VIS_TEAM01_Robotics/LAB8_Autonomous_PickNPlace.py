import sys
import time
from pathlib import Path
import numpy as np
import cv2
import cv2.aruco as aruco
import rtde_receive
import rtde_control
import rtde_io

# =============================================================================
# 1. CONFIGURACIÓN DE HARDWARE Y PARÁMETROS CALIBRADOS DE LA CELDA UR3e
# =============================================================================
ROBOT_IP = "10.110.20.4"
CAMERA_INDEX = 0
ROOT = Path(__file__).resolve().parent

WORKPIECE_MARKER_LENGTH = 0.035  # Marcadores ArUco en piezas: 3.5 cm (0.035 m)
PIECE_DIAMETER = 0.038           # Diámetro cilindro: 38 mm
PIECE_HEIGHT = 0.025             # Altura cilindro: 25 mm

# Alturas Z en marco Base {0} (en metros)
Z_APPROACH = 0.090   # 90 mm: paso libre seguro sobre piezas de 25 mm
Z_GRASP = 0.0056     # 5.6 mm: corresponde a -393.80 mm en el marco Local del pendant
Z_PLACE = 0.0075     # 7.5 mm: suelta suave sobre la mesa sin golpear
Z_MIN = 0.000        # Límite inferior TCP (0 mm)
Z_MAX = 0.250        # Límite superior TCP (250 mm)

# TCP calibrado final [+19 mm en X_tool, -41 mm en Y_tool, +108.5 mm en Z_tool]
TCP = [0.019, -0.041, 0.1085, 0.0, 0.0, 0.0]

PAYLOAD_EMPTY = (0.56, [0.0, 0.0, 0.054])
PAYLOAD_HOLDING = (0.60, [0.005, -0.010, 0.056])

# Postura segura inicial y parámetros DH nominales UR3e
Q_SAFE = np.deg2rad([0, -90, 90, -90, -90, 0])
A = [0.0, -0.24355, -0.21320, 0.0, 0.0, 0.0]
D = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.09210]
ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]

# Geolímites XY del centro de las mordazas (límite +X ampliado a 0.210 m para permitir depósito en 0.195 m)
BOUNDS = (-0.19741, 0.21000, -0.39620, -0.14971)
MARGIN = 0.010
MU_MIN = 0.005       # Manipulabilidad mínima permitida (m^3)
V_MAX = 0.025        # Velocidad vertical máxima speedL: 25 mm/s (< 50 mm/s Lab08)
SPEED_ACCEL = 0.05
CONTROL_DT = 0.02
RTDE_FREQ = 125.0    # Frecuencia RTDE estable en Windows (suficiente para CONTROL_DT = 0.02 s)

NAMES = {
    10: "WP 1 (Blanca)",
    11: "WP 2 (Morada)",
    12: "WP 3 (Roja)"
}

# Compensación XY en el punto de agarre por pieza [delta_X, delta_Y] (en metros):
# - ID 10 (Blanca): +12.00 mm en X y +1.00 mm en Y -> lleva [-0.140, -0.274] m a [-0.128, -0.273] m.
# - ID 11 (Morada): -0.91 mm (-0.00091 m) en X y -11.30 mm (-0.01130 m) en Y -> [41.77, -251.55] mm.
# - ID 12 (Roja):   sin compensación adicional.
PICK_OFFSET_XY = {
    10: np.array([0.01200, 0.00100]),
    11: np.array([-0.00091, -0.01130]),
    12: np.array([0.00000, 0.00000])
}

# Puntos de depósito [X, Y] en Base {0} (metros):
DROP_TARGETS = {
    10: np.array([0.195, -0.215]),  # Destino Blanca (ID 10)
    11: np.array([0.195, -0.285]),  # Destino Morada (ID 11)
    12: np.array([0.195, -0.355])   # Destino Roja   (ID 12)
}


# =============================================================================
# 2. CINEMÁTICA CON TCP CALIBRADO 3D, JACOBIANO Y VERIFICACIÓN DE SEGURIDAD
# =============================================================================
def inside_xy(p):
    p = np.asarray(p)
    return (
        p.shape[0] >= 2
        and np.isfinite(p[:2]).all()
        and (BOUNDS[0] + MARGIN) <= p[0] <= (BOUNDS[1] - MARGIN)
        and (BOUNDS[2] + MARGIN) <= p[1] <= (BOUNDS[3] - MARGIN)
    )


def fk_flange(q):
    T = np.eye(4)
    for th, a, d, al in zip(q, A, D, ALPHA):
        c, s, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        T = T @ np.array([
            [c, -s * ca,  s * sa, a * c],
            [s,  c * ca, -c * sa, a * s],
            [0.0, sa,     ca,     d],
            [0.0, 0.0,    0.0,    1.0]
        ])
    return T


def fk(q):
    T = fk_flange(q)
    T[:3, 3] += T[:3, :3] @ np.array(TCP[:3])
    return T


def ik_tcp(p_tcp):
    """Cinemática inversa para el centro real entre mordazas con orientación fija."""
    p_tcp = np.asarray(p_tcp, dtype=float)
    p_flange = p_tcp + np.array([0.0, 0.0, TCP[2]])
    q = None
    for _ in range(6):
        x, y, z = p_flange
        r2 = x * x + y * y
        if r2 < D[3] ** 2:
            return None
        arm = -np.sqrt(r2 - D[3] ** 2)
        q1 = np.arctan2((D[3] * x + arm * y) / r2, (arm * x - D[3] * y) / r2)
        rw, sw = arm + D[4], z - D[0] + D[5]
        c3 = (rw * rw + sw * sw - A[1] ** 2 - A[2] ** 2) / (2.0 * A[1] * A[2])
        if abs(c3) > 1.0:
            return None
        s3 = np.sqrt(1.0 - c3 * c3)
        q3 = np.arctan2(s3, c3)
        q2 = np.arctan2(sw, rw) - np.arctan2(A[2] * s3, A[1] + A[2] * c3)
        q4 = -np.pi / 2.0 - q2 - q3
        q5 = -np.pi / 2.0
        q6 = q1  # Mantiene orientación constante de la garra respecto a Base
        q = np.array([q1, q2, q3, q4, q5, q6])
        R = fk_flange(q)[:3, :3]
        p_flange = p_tcp - R @ np.array(TCP[:3])
    return np.arctan2(np.sin(q), np.cos(q))


def jacobian_tcp(q):
    T = np.eye(4)
    origins, axes = [], []
    for th, a, d, al in zip(q, A, D, ALPHA):
        origins.append(T[:3, 3].copy())
        axes.append(T[:3, 2].copy())
        c, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        T = T @ np.array([
            [c,  -st * ca,  st * sa, a * c],
            [st,  c * ca,  -c * sa,  a * st],
            [0.0, sa,       ca,      d],
            [0.0, 0.0,      0.0,     1.0]
        ])
    p = T[:3, 3] + T[:3, :3] @ np.array(TCP[:3])
    Jv = np.column_stack([np.cross(z, p - o) for z, o in zip(axes, origins)])
    return np.vstack((Jv, np.column_stack(axes)))


def manipulability(q):
    return float(np.prod(np.linalg.svd(jacobian_tcp(q)[:3], compute_uv=False)))


def nearest(q, current):
    return current + np.arctan2(np.sin(q - current), np.cos(q - current))


def validate_vertical_trajectory(x, y, z_high, z_low, q_ref, rtde_c):
    """Valida IK, FK, manipulabilidad y límites del controlador antes de mover."""
    if not inside_xy([x, y]):
        raise ValueError(f"Punto TCP XY [{x*1000:.1f}, {y*1000:.1f}] mm fuera de geolímites.")
    poses, q_high = [], None
    for z in np.linspace(z_high, z_low, 15):
        tcp_target = np.array([x, y, z])
        q = ik_tcp(tcp_target)
        if q is None:
            raise ValueError(f"IK sin solución en [{x:.3f}, {y:.3f}, {z:.3f}].")
        if manipulability(q) < MU_MIN:
            raise ValueError(f"Manipulabilidad baja (< {MU_MIN}) en [{x:.3f}, {y:.3f}, {z:.3f}].")
        T = fk(q)
        if np.linalg.norm(T[:3, 3] - tcp_target) > 0.0001:
            raise ValueError("Error de comprobación IK/FK > 0.1 mm.")
        rv = cv2.Rodrigues(T[:3, :3])[0].ravel()
        pose = np.r_[tcp_target, rv]
        if not rtde_c.isPoseWithinSafetyLimits(pose.tolist()):
            raise ValueError(f"Pose rechazada por controlador UR3e en Z={z*1000:.1f} mm.")
        if q_high is None:
            q_high = nearest(q, q_ref)
        poses.append(pose)
    return q_high, poses[0], poses[-1]


# =============================================================================
# 3. VISIÓN ARTIFICIAL: DETECCIÓN ESTABLE EN PLANO XY (IDs 10, 11, 12)
# =============================================================================
def project_marker_center_to_plane(xy, K, dist, T_cam_to_base, z_plane):
    """Proyecta el centro geométrico 2D (intersección de diagonales) al plano Z en Base."""
    undist = cv2.undistortPoints(xy.reshape(-1, 1, 2).astype(np.float32), K, dist).reshape(4, 2)
    p0, p1, p2, p3 = undist
    d1 = p2 - p0
    d2 = p3 - p1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) > 1e-8:
        t = ((p1[0] - p0[0]) * d2[1] - (p1[1] - p0[1]) * d2[0]) / denom
        center_n = p0 + t * d1
    else:
        center_n = undist.mean(axis=0)

    ray_cam = np.array([center_n[0], center_n[1], 1.0], dtype=float)
    origin_base = T_cam_to_base[:3, 3]
    dir_base = T_cam_to_base[:3, :3] @ ray_cam
    if abs(dir_base[2]) < 1e-6:
        return None
    s = (z_plane - origin_base[2]) / dir_base[2]
    if s <= 0:
        return None
    return origin_base + s * dir_base


def detect_workpieces(K, dist, T_cam_to_base, robot=None):
    """Muestra vista previa y captura la mediana XY filtrada en Base de los IDs 10, 11 y 12."""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 75
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 5
    params.polygonalApproxAccuracyRate = 0.05
    params.maxErroneousBitsInBorderRate = 0.45
    params.minDistanceToBorder = 1
    params.minCornerDistanceRate = 0.01
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(dictionary, params)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    L = WORKPIECE_MARKER_LENGTH
    obj_pts = np.array([
        [-L / 2,  L / 2, 0.0],
        [ L / 2,  L / 2, 0.0],
        [ L / 2, -L / 2, 0.0],
        [-L / 2, -L / 2, 0.0]
    ], dtype=np.float32)

    print("\n[VISIÓN] Verifica en pantalla que las piezas estén dentro del área segura.")
    print("[VISIÓN] Presiona 's' o ESPACIO para iniciar el Pick & Place autónomo ('q' para cancelar).\n")

    samples = {mid: [] for mid in NAMES}
    capturing = False
    capture_start_t = 0.0
    target_samples = 20

    while True:
        if robot is not None:
            robot.keep_alive()

        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        detected_set = set(int(x) for x in ids.flatten()) if ids is not None else set()
        if not set(NAMES.keys()).issubset(detected_set):
            gray_enh = clahe.apply(gray)
            c_alt, ids_alt, _ = detector.detectMarkers(gray_enh)
            if ids_alt is not None:
                if ids is None:
                    corners = c_alt
                    ids = np.asarray(ids_alt, dtype=np.int32).reshape(-1, 1)
                else:
                    corners = list(corners)
                    ids_list = [int(x) for x in ids.flatten()]
                    for idx_a, id_v in enumerate(ids_alt.flatten()):
                        id_int = int(id_v)
                        if id_int not in detected_set:
                            corners.append(c_alt[idx_a])
                            ids_list.append(id_int)
                            detected_set.add(id_int)
                    ids = np.array(ids_list, dtype=np.int32).reshape(-1, 1)

        vis = frame.copy()
        frame_pts = {}

        if ids is not None:
            ids = np.asarray(ids, dtype=np.int32).reshape(-1, 1)
            aruco.drawDetectedMarkers(vis, corners, ids)
            flat_ids = ids.flatten()
            for i, raw_id in enumerate(flat_ids):
                mid = int(raw_id)
                if mid in NAMES and np.count_nonzero(flat_ids == mid) == 1:
                    xy = corners[i][0].astype(np.float32)
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_pts, xy, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if ok and tvec[2, 0] > 0:
                        proj = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)[0].reshape(-1, 2)
                        rms = float(np.sqrt(np.mean(np.sum((proj - xy) ** 2, axis=1))))
                        if rms <= 1.8:
                            p_pnp = (T_cam_to_base @ np.append(tvec.flatten(), 1.0))[:3]
                            # Descarta detecciones espurias con altura Z irreal respecto a la mesa
                            if not (-0.015 <= p_pnp[2] <= 0.075):
                                continue

                            if mid == 10:
                                p_plane = project_marker_center_to_plane(
                                    xy, K, dist, T_cam_to_base, z_plane=PIECE_HEIGHT
                                )
                                p_base = p_plane if p_plane is not None else p_pnp
                            else:
                                p_base = p_pnp

                            frame_pts[mid] = p_base
                            off = PICK_OFFSET_XY.get(mid, np.zeros(2))
                            p_pick_vis = p_base[:2] + off
                            ok_xy = inside_xy(p_pick_vis)
                            color = (0, 255, 0) if ok_xy else (0, 0, 255)
                            pt_txt = tuple(xy[0].astype(int))
                            cv2.putText(
                                vis,
                                f"ID {mid}: [{p_pick_vis[0]*1000:.0f}, {p_pick_vis[1]*1000:.0f}] mm",
                                (pt_txt[0], max(22, pt_txt[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                            )

        if not capturing:
            msg = f"Detectados: {sorted(frame_pts.keys())} | Presiona ESPACIO/'s' para iniciar"
        else:
            if time.monotonic() - capture_start_t >= 0.20:
                for mid, p in frame_pts.items():
                    off = PICK_OFFSET_XY.get(mid, np.zeros(2))
                    if inside_xy(p[:2] + off) and len(samples[mid]) < target_samples:
                        samples[mid].append(p)
            counts = {mid: len(samples[mid]) for mid in NAMES}
            msg = f"Capturando muestras: {counts}"
            active_ids = [mid for mid, cnt in counts.items() if cnt > 0]
            if active_ids and all(counts[mid] >= target_samples for mid in active_ids):
                break

        cv2.putText(vis, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("Fase C: Autonomous Pick & Place", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord(" "), ord("s")) and not capturing:
            if not frame_pts:
                print("[AVISO] No hay ninguna pieza (10, 11 o 12) visible en pantalla.")
            else:
                samples = {mid: [] for mid in NAMES}
                capture_start_t = time.monotonic()
                capturing = True
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    if robot is not None:
        robot.keep_alive()

    detected_base = {}
    for mid in [10, 11, 12]:
        if len(samples[mid]) >= 10:
            pts = np.array(samples[mid])
            med_raw = np.median(pts, axis=0)
            dists_mm = np.linalg.norm(pts[:, :2] - med_raw[:2], axis=1) * 1000.0
            inliers = pts[dists_mm <= 5.0]
            if len(inliers) < 10:
                raise ValueError(
                    f"Detección XY inestable en ID {mid} (solo {len(inliers)}/20 muestras consistentes)."
                )
            med = np.median(inliers, axis=0)
            spread_xy_mm = float(np.max(np.linalg.norm(inliers[:, :2] - med[:2], axis=1)) * 1000.0)
            print(
                f"[VISIÓN] {NAMES[mid]} (ID {mid}): XY=[{med[0]*1000:.1f}, {med[1]*1000:.1f}] mm "
                f"(dispersión inliers={spread_xy_mm:.2f} mm, {len(inliers)}/20)"
            )
            detected_base[mid] = med
    return detected_base


# =============================================================================
# 4. CONTROL DE MOVIMIENTO UR3e (MoveJ + speedL CERRADO EN Z)
# =============================================================================
class UR3eController:
    def __init__(self, ip):
        self.ip = ip
        self.r = None
        self.c = None
        self.io = None
        self._connect_all()
        self.check_safety()

    def _connect_receive(self):
        try:
            return rtde_receive.RTDEReceiveInterface(self.ip, RTDE_FREQ)
        except TypeError:
            return rtde_receive.RTDEReceiveInterface(self.ip)

    def _connect_all(self):
        self.r = self._connect_receive()
        self.c = rtde_control.RTDEControlInterface(self.ip)
        self.io = rtde_io.RTDEIOInterface(self.ip)
        self.c.setTcp(TCP)

    def _ensure_connected(self):
        if self.r is None or not self.r.isConnected():
            try:
                if self.r is not None:
                    self.r.disconnect()
            except Exception:
                pass
            time.sleep(0.15)
            self.r = self._connect_receive()

        if self.c is None or not self.c.isConnected():
            try:
                if self.c is not None:
                    self.c.disconnect()
            except Exception:
                pass
            time.sleep(0.15)
            self.c = rtde_control.RTDEControlInterface(self.ip)
            self.c.setTcp(TCP)

    def keep_alive(self):
        self._ensure_connected()
        _ = self.r.getActualQ()

    def check_safety(self):
        self._ensure_connected()
        if not self.r.isConnected() or not self.c.isConnected():
            raise RuntimeError("Conexión RTDE perdida.")
        if self.r.isProtectiveStopped() or self.r.isEmergencyStopped():
            raise RuntimeError("Robot detenido por parada de protección o emergencia.")

    def move_j(self, q_target):
        self.check_safety()
        q_target = np.asarray(q_target, dtype=float)
        if not self.c.isJointsWithinSafetyLimits(q_target.tolist()):
            raise ValueError("Objetivo articular fuera de límites de seguridad.")
        if not self.c.moveJ(q_target.tolist(), 0.25, 0.30, True):
            raise RuntimeError("Comando moveJ rechazado.")
        start, settled = time.monotonic(), None
        while time.monotonic() - start < 60.0:
            self.check_safety()
            q_act = np.asarray(self.r.getActualQ())
            qd_act = np.asarray(self.r.getActualQd())
            if np.max(np.abs(q_act - q_target)) < 0.003 and np.max(np.abs(qd_act)) < 0.002:
                if settled is None:
                    settled = time.monotonic()
                elif time.monotonic() - settled >= 0.25:
                    self.c.stopJ(1.0)
                    time.sleep(0.05)
                    return
            else:
                settled = None
            time.sleep(0.02)
        self.c.stopJ(0.5)
        raise RuntimeError("Tiempo excedido en moveJ.")

    def move_vertical_z(self, z_target):
        """Desplaza únicamente en Z manteniendo fijo el XY real alcanzado tras move_j."""
        self.check_safety()
        real_start = np.asarray(self.r.getActualTCPPose(), dtype=float)
        target_pose = real_start.copy()
        target_pose[2] = float(z_target)

        if not inside_xy(target_pose[:3]) or not (Z_MIN <= target_pose[2] <= Z_MAX):
            raise ValueError("Objetivo vertical fuera de geolímites.")

        start = time.monotonic()
        started = False
        try:
            while time.monotonic() - start < 45.0:
                self.check_safety()
                now = time.monotonic()
                pose = np.asarray(self.r.getActualTCPPose(), dtype=float)
                q = np.asarray(self.r.getActualQ(), dtype=float)
                if np.linalg.norm(pose[:2] - target_pose[:2]) > 0.0025:
                    raise RuntimeError("Deriva XY > 2.5 mm durante movimiento vertical.")
                if manipulability(q) < MU_MIN:
                    raise RuntimeError("Manipulabilidad por debajo del umbral durante speedL.")
                err_z = float(target_pose[2] - pose[2])
                if abs(err_z) <= 0.0005:
                    break
                vz = float(np.clip(1.8 * err_z, -V_MAX, V_MAX))
                started = True
                if not self.c.speedL([0.0, 0.0, vz, 0.0, 0.0, 0.0], SPEED_ACCEL, CONTROL_DT):
                    raise RuntimeError("speedL rechazado.")
                time.sleep(max(0.0, CONTROL_DT - (time.monotonic() - now)))
            else:
                raise RuntimeError("Tiempo excedido en descenso/ascenso vertical.")
        finally:
            if started:
                self.c.speedStop(0.25)
                time.sleep(0.05)

    def set_gripper(self, close: bool):
        """Acciona la garra: DO1=False -> DO0=True abre; DO0=False -> DO1=True cierra."""
        self.check_safety()
        if close:
            ok0 = self.io.setStandardDigitalOut(0, False)
            time.sleep(0.1)
            ok1 = self.io.setStandardDigitalOut(1, True)
        else:
            ok1 = self.io.setStandardDigitalOut(1, False)
            time.sleep(0.1)
            ok0 = self.io.setStandardDigitalOut(0, True)

        if not (ok0 and ok1):
            raise RuntimeError("No se pudieron accionar las salidas digitales (DO0/DO1) de la garra.")

        time.sleep(0.6)

        bits = self.r.getActualDigitalOutputBits()
        do0 = (bits >> 0) & 1
        do1 = (bits >> 1) & 1
        esperado = (0, 1) if close else (1, 0)
        if (do0, do1) != esperado:
            if close:
                self.io.setStandardDigitalOut(0, False)
                time.sleep(0.1)
                self.io.setStandardDigitalOut(1, True)
            else:
                self.io.setStandardDigitalOut(1, False)
                time.sleep(0.1)
                self.io.setStandardDigitalOut(0, True)
            time.sleep(0.6)

        mass, com = PAYLOAD_HOLDING if close else PAYLOAD_EMPTY
        self.c.setPayload(mass, com)

    def disconnect(self):
        for iface in (self.io, self.c, self.r):
            try:
                if iface is not None:
                    iface.disconnect()
            except Exception:
                pass


# =============================================================================
# 5. EJECUCIÓN AUTÓNOMA DE PICK & PLACE
# =============================================================================
if __name__ == "__main__":
    try:
        with np.load(ROOT / "camera_intrinsics.npz", allow_pickle=False) as data:
            K = np.asarray(data["camera_matrix"], dtype=np.float32)
            dist = np.asarray(data["dist_coeffs"], dtype=np.float32)
        T_cam_to_base = np.load(ROOT / "T_cam_to_base.npy")
    except FileNotFoundError as e:
        print(f"[ERROR] Falta archivo de calibración ({e.filename}).")
        sys.exit(1)

    for mid, xy_drop in DROP_TARGETS.items():
        if not inside_xy(xy_drop):
            print(f"[ERROR] Destino de ID {mid} ({xy_drop}) fuera de geolímites XY.")
            sys.exit(1)

    robot = None
    try:
        print(f"[ROBOT] Conectando a UR3e en {ROBOT_IP}...")
        robot = UR3eController(ROBOT_IP)
        robot.c.setTcp(TCP)
        robot.set_gripper(close=False)

        print("[ROBOT] Moviendo a Q_SAFE para despejar vista de cámara...")
        q_home = nearest(Q_SAFE, np.asarray(robot.r.getActualQ()))
        robot.move_j(q_home)

        detected_pieces = detect_workpieces(K, dist, T_cam_to_base, robot=robot)
        if not detected_pieces:
            print("[AVISO] No se detectaron piezas válidas. Saliendo.")
            sys.exit(0)

        print("\n" + "=" * 95)
        print("                 PLAN DE EJECUCIÓN AUTÓNOMA (PICK & PLACE)")
        print("=" * 95)
        for mid, p_pick in detected_pieces.items():
            offset_xy = PICK_OFFSET_XY.get(mid, np.zeros(2))
            px_plan = float(p_pick[0] + offset_xy[0])
            py_plan = float(p_pick[1] + offset_xy[1])
            p_drop = DROP_TARGETS[mid]
            print(
                f"{NAMES[mid]:<16} (ID {mid}) | "
                f"Pick XY: [{px_plan:+.3f}, {py_plan:+.3f}] m  -->  "
                f"Place XY: [{p_drop[0]:+.3f}, {p_drop[1]:+.3f}] m"
            )
        print("=" * 95 + "\n")

        for mid, p_pick in detected_pieces.items():
            offset_xy = PICK_OFFSET_XY.get(mid, np.zeros(2))
            px = float(p_pick[0] + offset_xy[0])
            py = float(p_pick[1] + offset_xy[1])
            dx, dy = float(DROP_TARGETS[mid][0]), float(DROP_TARGETS[mid][1])

            if np.hypot(px - dx, py - dy) < (PIECE_DIAMETER + 0.015):
                print(f"[SALTO] {NAMES[mid]} ya está sobre su zona de depósito.")
                continue

            print(f"\n>>> Procesando {NAMES[mid]} (ID {mid})...")

            q_curr = np.asarray(robot.r.getActualQ())
            q_pick_app, _, _ = validate_vertical_trajectory(
                px, py, Z_APPROACH, Z_GRASP, q_curr, robot.c
            )
            q_drop_app, _, _ = validate_vertical_trajectory(
                dx, dy, Z_APPROACH, Z_PLACE, q_pick_app, robot.c
            )

            print(f"  1. Aproximando mordazas a [{px*1000:.1f}, {py*1000:.1f}, {Z_APPROACH*1000:.0f}] mm...")
            robot.move_j(q_pick_app)

            print(f"  2. Descendiendo a Z_agarre = {Z_GRASP*1000:.1f} mm...")
            robot.move_vertical_z(Z_GRASP)

            print("  3. Cerrando garra...")
            robot.set_gripper(close=True)

            print("  4. Elevando pieza...")
            robot.move_vertical_z(Z_APPROACH)

            print(f"  5. Trasladando a destino [{dx*1000:.1f}, {dy*1000:.1f}, {Z_APPROACH*1000:.0f}] mm...")
            robot.move_j(q_drop_app)

            print(f"  6. Descendiendo a Z_depósito = {Z_PLACE*1000:.1f} mm...")
            robot.move_vertical_z(Z_PLACE)

            print("  7. Soltando pieza...")
            robot.set_gripper(close=False)

            print("  8. Retirada vertical...")
            robot.move_vertical_z(Z_APPROACH)

        print("\n[ROBOT] Ciclo completo. Regresando a Q_SAFE...")
        robot.move_j(nearest(Q_SAFE, np.asarray(robot.r.getActualQ())))
        print("[OK] Fase C (Autonomous Pick & Place) finalizada con éxito.")

    except Exception as err:
        print(f"\n[ABORTADO] {err}")
    finally:
        if robot is not None:
            robot.disconnect()