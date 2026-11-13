import sys
import cv2
import cv2.aruco as aruco
import numpy as np

CAMERA_INDEX = 0
WORKPIECE_MARKER_LENGTH = 0.035  # Marcadores de las piezas: 3.5 cm (0.035 m)

# 1. Cargar intrínsecos reales y matriz extrínseca T_cam_to_base de la Fase A
try:
    with np.load("camera_intrinsics.npz", allow_pickle=False) as data:
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float32)
    T_cam_to_base = np.load("T_cam_to_base.npy")
    print("[INFO] 'camera_intrinsics.npz' y 'T_cam_to_base.npy' cargados correctamente.")
except FileNotFoundError as e:
    print(f"[ERROR] Falta archivo de calibración ({e.filename}). Ejecuta primero la Fase A.")
    sys.exit(1)

# Mapa ajustado a tus piezas físicas (ID 10 Blanca, ID 11 Morada, ID 12 Roja)
WORKPIECE_MAP = {
    10: "WP 1 (Blanca / ID 10)",
    11: "WP 2 (Morada / ID 11)",
    12: "WP 3 (Roja / ID 12)"
}


def estimate_marker_pose_3d(corners, marker_length, K, dist):
    obj_points = np.array([
        [-marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ], dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(
        obj_points, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if success and tvec[2, 0] > 0:
        return tvec.flatten(), rvec.flatten()
    return None, None


def transform_cam_to_base(p_cam, T_matrix):
    p_base_homo = T_matrix @ np.append(p_cam, 1.0)
    return p_base_homo[:3]


def print_table_8_2(detected_data):
    print("\n" + "=" * 115)
    print("                           TABLA 8.2: MULTI-OBJECT 3D LOCALIZATION & BASE FRAME MAPPING")
    print("=" * 115)
    print(f"{'Workpiece':<22} | {'ArUco ID':<10} | {'Detected Cam [xc, yc, zc] (m)':<30} | {'Transformed Base [x0, y0, z0] (m)':<33} | {'Physical Meas. (m)'}")
    print("-" * 115)

    for marker_id in [10, 11, 12]:
        if marker_id in detected_data:
            p_cam, p_base = detected_data[marker_id]
            cam_str = f"[{p_cam[0]:+.3f}, {p_cam[1]:+.3f}, {p_cam[2]:+.3f}]"
            base_str = f"[{p_base[0]:+.3f}, {p_base[1]:+.3f}, {p_base[2]:+.3f}]"
            print(f"{WORKPIECE_MAP[marker_id]:<22} | ID {marker_id:<7} | {cam_str:<30} | {base_str:<33} | [           ]")
        else:
            print(f"{WORKPIECE_MAP[marker_id]:<22} | ID {marker_id:<7} | {'[ NO DETECTADO ]':<30} | {'[ NO DETECTADO ]':<33} | [           ]")
    print("=" * 115 + "\n")


if __name__ == '__main__':
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo acceder a la webcam.")
        sys.exit(1)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 65
    parameters.adaptiveThreshWinSizeStep = 2
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.maxErroneousBitsInBorderRate = 0.6
    parameters.minDistanceToBorder = 1
    parameters.minCornerDistanceRate = 0.01
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

    detector = aruco.ArucoDetector(dictionary, parameters)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print("\n[INFO] Rastreo 3D activado. Coloca las piezas 10, 11 y 12 dentro del área de trabajo.")
    print("[INFO] Presiona 'c' para imprimir la TABLA 8.2 | Presiona 'q' para salir.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) < 3:
            gray_enhanced = clahe.apply(gray)
            corners_alt, ids_alt, _ = detector.detectMarkers(gray_enhanced)
            if ids_alt is not None:
                if ids is None:
                    corners, ids = corners_alt, ids_alt
                else:
                    existing_ids = list(ids.flatten())
                    corners = list(corners)
                    for idx_alt, id_val in enumerate(ids_alt.flatten()):
                        if id_val not in existing_ids:
                            corners.append(corners_alt[idx_alt])
                            ids = np.append(ids, [[id_val]], axis=0)

        current_detected = {}
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_id in enumerate(ids.flatten()):
                mid = int(marker_id)
                if mid in WORKPIECE_MAP:
                    p_cam, rvec = estimate_marker_pose_3d(
                        corners[i][0], WORKPIECE_MARKER_LENGTH, camera_matrix, dist_coeffs
                    )
                    if p_cam is not None:
                        p_base = transform_cam_to_base(p_cam, T_cam_to_base)
                        current_detected[mid] = (p_cam, p_base)

                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, p_cam, 0.03)

                        label_cam = f"Cam: [{p_cam[0]:+.3f}, {p_cam[1]:+.3f}, {p_cam[2]:+.3f}] m"
                        label_base = f"Base: [{p_base[0]:+.3f}, {p_base[1]:+.3f}, {p_base[2]:+.3f}] m"

                        corner_pt = tuple(corners[i][0][0].astype(int))
                        cv2.putText(frame, WORKPIECE_MAP[mid], (corner_pt[0], max(20, corner_pt[1] - 32)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                        cv2.putText(frame, label_cam, (corner_pt[0], max(38, corner_pt[1] - 16)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                        cv2.putText(frame, label_base, (corner_pt[0], max(54, corner_pt[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow("Fase B: Localizacion 3D de Objetos ('c'=Tabla 8.2 | 'q'=Salir)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print_table_8_2(current_detected)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()