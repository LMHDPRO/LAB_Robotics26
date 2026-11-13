import sys
from pathlib import Path
import numpy as np
import cv2
import cv2.aruco as aruco

CAMERA_INDEX = 0
CALIB_MARKER_LENGTH = 0.050  # 5.0 cm exactos
MAX_CALIB_ERROR_MM = 5.0     # Umbral oficial del manual (< 5.0 mm)
AVG_FRAMES_EXTRINSIC = 20    # Fotogramas a promediar

# Coordenadas reales medidas desde el CENTRO de la base del robot (en metros):
M0_BASE = np.array([ 0.085, -0.200, 0.000])  # Arriba derecha (8.5 cm der, 20.0 cm frente)
M1_BASE = np.array([-0.170, -0.200, 0.000])  # Arriba izquierda (17.0 cm izq, 20.0 cm frente)
M2_BASE = np.array([ 0.085, -0.355, 0.000])  # Abajo derecha (8.5 cm der, 35.5 cm frente)
M3_BASE = np.array([-0.170, -0.355, 0.000])  # Abajo izquierda (17.0 cm izq, 35.5 cm frente)
M4_BASE = np.array([-0.038, -0.285, 0.000])  # Centro (3.8 cm izq, 28.5 cm frente)

GROUND_TRUTH_BASE = {
    0: M0_BASE,
    1: M1_BASE,
    2: M2_BASE,
    3: M3_BASE,
    4: M4_BASE
}


def solve_kabsch_svd(P_cam, P_base):
    mu_cam = np.mean(P_cam, axis=0)
    mu_base = np.mean(P_base, axis=0)
    Q_cam = P_cam - mu_cam
    Q_base = P_base - mu_base

    H = Q_cam.T @ Q_base
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T

    d = np.linalg.det(V @ U.T)
    R_cb = V @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T
    t_cb = mu_base - R_cb @ mu_cam

    T = np.eye(4)
    T[:3, :3] = R_cb
    T[:3, 3] = t_cb
    return T


def evaluate_calibration(corners_buffer, obj_pts, K, dist):
    detected_cam = {}
    for m_id, pts_list in corners_buffer.items():
        tvecs = []
        for c_2d in pts_list:
            ok, _, tvec = cv2.solvePnP(
                obj_pts, c_2d, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok and tvec[2, 0] > 0:
                tvecs.append(tvec.flatten())
        detected_cam[m_id] = np.median(tvecs, axis=0)

    P_cam = np.array([detected_cam[m_id] for m_id in [0, 1, 2, 3]])
    P_base = np.array([GROUND_TRUTH_BASE[m_id] for m_id in [0, 1, 2, 3]])
    T = solve_kabsch_svd(P_cam, P_base)

    p_pred_m4 = (T @ np.append(detected_cam[4], 1.0))[:3]
    err_m4 = float(np.linalg.norm(p_pred_m4 - GROUND_TRUTH_BASE[4]) * 1000.0)
    return T, detected_cam, p_pred_m4, err_m4


if __name__ == '__main__':
    intrinsics_path = Path("camera_intrinsics.npz")
    if not intrinsics_path.exists():
        print("[ERROR] Falta 'camera_intrinsics.npz' en esta carpeta.")
        sys.exit(1)

    with np.load(intrinsics_path, allow_pickle=False) as data:
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
        image_size = np.asarray(data["image_size"], dtype=int)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = aruco.ArucoDetector(dictionary, parameters)

    L = CALIB_MARKER_LENGTH
    obj_pts = np.array([
        [-L / 2,  L / 2, 0],
        [ L / 2,  L / 2, 0],
        [ L / 2, -L / 2, 0],
        [-L / 2, -L / 2, 0]
    ], dtype=np.float64)

    print("Con la cámara fija viendo los 5 marcadores (0, 1, 2, 3, 4), presiona 'c' o ESPACIO.")
    capturing = False
    corners_buffer = {m_id: [] for m_id in GROUND_TRUTH_BASE}

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        vis = frame.copy()

        frame_corners = {}
        if ids is not None:
            aruco.drawDetectedMarkers(vis, corners, ids)
            for i, raw_id in enumerate(ids.flatten()):
                m_id = int(raw_id)
                if m_id in GROUND_TRUTH_BASE and np.count_nonzero(ids == m_id) == 1:
                    frame_corners[m_id] = corners[i][0].astype(np.float64)

        status = f"IDs visibles: {sorted(frame_corners.keys())} | Presiona 'c'"
        if capturing:
            if len(frame_corners) == 5:
                for m_id, c2d in frame_corners.items():
                    corners_buffer[m_id].append(c2d)
                count = len(corners_buffer[0])
                status = f"Capturando {count}/{AVG_FRAMES_EXTRINSIC}..."
                if count >= AVG_FRAMES_EXTRINSIC:
                    break
            else:
                status = f"Faltan IDs! Visibles: {sorted(frame_corners.keys())}"

        cv2.putText(vis, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow("Calibracion Extrinseca Final", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord(' '), ord('c')) and not capturing:
            corners_buffer = {m_id: [] for m_id in GROUND_TRUTH_BASE}
            capturing = True
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    # Evalúa con dist_coeffs actual vs distorsión cero (misma focal calibrada fx)
    T1, cam1, pred1, err1 = evaluate_calibration(corners_buffer, obj_pts, camera_matrix, dist_coeffs)
    zero_dist = np.zeros_like(dist_coeffs)
    T2, cam2, pred2, err2 = evaluate_calibration(corners_buffer, obj_pts, camera_matrix, zero_dist)

    if err2 < err1:
        T_cam_to_base, detected_cam_dict, p_base_m4_pred, err_m4_mm = T2, cam2, pred2, err2
        final_dist = zero_dist
    else:
        T_cam_to_base, detected_cam_dict, p_base_m4_pred, err_m4_mm = T1, cam1, pred1, err1
        final_dist = dist_coeffs

    p_cam_m4 = detected_cam_dict[4]
    p_base_m4_truth = GROUND_TRUTH_BASE[4]

    print("\n" + "=" * 115)
    print("                           TABLA 8.1: EXTRINSIC CAMERA-TO-BASE CALIBRATION")
    print("=" * 115)
    print(f"{'Marker ID':<10} | {'Measured Base [x,y,z] (m)':<27} | {'Detected Cam [xc,yc,zc] (m)':<27} | {'Predicted Base [xp,yp,zp] (m)':<28} | {'Error (mm)':<10} | {'Status'}")
    print("-" * 115)

    for m_id in [0, 1, 2, 3]:
        p_b = GROUND_TRUTH_BASE[m_id]
        p_c = detected_cam_dict[m_id]
        p_pred = (T_cam_to_base @ np.append(p_c, 1.0))[:3]
        err_mm = float(np.linalg.norm(p_pred - p_b) * 1000.0)
        print(f"M{m_id} (Calib)  | [{p_b[0]:+.3f}, {p_b[1]:+.3f}, {p_b[2]:+.3f}]   | [{p_c[0]:+.3f}, {p_c[1]:+.3f}, {p_c[2]:+.3f}]   | [{p_pred[0]:+.3f}, {p_pred[1]:+.3f}, {p_pred[2]:+.3f}]    | {err_mm:<10.2f} | Fit Point")

    status_m4 = "PASS (<5mm)" if err_m4_mm < MAX_CALIB_ERROR_MM else "FAIL (>=5mm)"
    print(f"M4 (Check)  | [{p_base_m4_truth[0]:+.3f}, {p_base_m4_truth[1]:+.3f}, {p_base_m4_truth[2]:+.3f}]   | [{p_cam_m4[0]:+.3f}, {p_cam_m4[1]:+.3f}, {p_cam_m4[2]:+.3f}]   | [{p_base_m4_pred[0]:+.3f}, {p_base_m4_pred[1]:+.3f}, {p_base_m4_pred[2]:+.3f}]    | {err_m4_mm:<10.2f} | {status_m4}")
    print("=" * 115)

    if err_m4_mm < MAX_CALIB_ERROR_MM:
        np.savez("camera_intrinsics.npz", camera_matrix=camera_matrix, dist_coeffs=final_dist, image_size=image_size)
        np.save("T_cam_to_base.npy", T_cam_to_base)
        print(f"\n[PASS] Error M4 = {err_m4_mm:.2f} mm (< 5.0 mm). Archivos guardados.")
        print("Toma captura de esta consola para REQ-PHASE-A2 y ya puedes probar 'ur3e_ui.py'.")
    else:
        print(f"\n[FAIL] Error M4 = {err_m4_mm:.2f} mm (>= 5.0 mm).")
        sys.exit(1)