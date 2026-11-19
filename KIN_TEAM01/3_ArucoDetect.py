# aruco_detect.py -- Phase C: Front-Camera 3-Pose Tool Flange Extraction (Deliverable D3)
# Team: KIN_TeamXX | Lab 5, Phase C
import cv2
import numpy as np
import rtde_receive
import os
import time

ROBOT_IP = "169.254.15.34"  # IP )
CAM_INDEX = 0
MARKER_ID = 0
MARKER_SIZE = 0.050 # [m] Cuadro negro exterior de 50.0 mm
BRACKET_THICKNESS = 0.005 # [m] Grosor d_b de la brida al marcador (0.0 si está al ras)

# 1. Verificar existencia de archivos de calibración
if not os.path.exists("camera_K.npy") or not os.path.exists("camera_dist.npy"):
    raise FileNotFoundError("No se encontraron 'camera_K.npy' o 'camera_dist.npy'. Ejecuta primero 'step1_calibrate_or_fallback.py'.")

K = np.load("camera_K.npy")
dist = np.load("camera_dist.npy")

# 2. Inicializar detector ArUco (OpenCV 4.8+)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
h = MARKER_SIZE / 2.0
obj_pts = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

# Matriz de transformación de la cámara frontal al sistema Base {0} del robot
R_cam_to_base = np.array([
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0]
])

def measure_flange_and_rtde(cap, rtde_r):
    flange_samples = []
    attempts = 0
    max_attempts = 150  # Límite de intentos para evitar congelamientos

    print("Capturando muestras del marcador ArUco...")
    while len(flange_samples) < 20 and attempts < max_attempts:
        ret, frame = cap.read()
        attempts += 1
        if not ret:
            time.sleep(0.03)
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and MARKER_ID in ids.flatten():
            idx = list(ids.flatten()).index(MARKER_ID)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, corners[idx][0], K, dist)
            if ok:
                R_cam, _ = cv2.Rodrigues(rvec)
                p_flange_cam = tvec.ravel() - (R_cam @ np.array([0.0, 0.0, BRACKET_THICKNESS]))
                flange_samples.append(p_flange_cam)
        time.sleep(0.02)

    if len(flange_samples) < 20:
        raise RuntimeError(f"No se detectó el marcador ID {MARKER_ID} suficientes veces (solo {len(flange_samples)}/20 muestras). Verifica iluminación y orientación.")

    p_cam_mean = np.mean(flange_samples, axis=0)
    p_rtde = np.array(rtde_r.getActualTCPPose()[:3])
    return p_cam_mean, p_rtde

# 3. Verificación de Hardware y Conexiones
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo acceder a la cámara en CAM_INDEX = {CAM_INDEX}. Prueba cambiar CAM_INDEX a 1 o 2.")

print(f"Conectando al robot UR3e en {ROBOT_IP}...")
try:
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
except Exception as e:
    cap.release()
    raise ConnectionError(f"Fallo al conectar con el robot en {ROBOT_IP}. Verifica el cable Ethernet y la IP. Error original: {e}")

try:
    p_cam_ref, p_rtde_ref = None, None
    for cfg_name in ["C1 (Reference)", "C3", "C5"]:
        input(f"\nMueve el robot a {cfg_name} y presiona ENTER para capturar...")
        p_cam, p_rtde = measure_flange_and_rtde(cap, rtde_r)
        
        if p_cam_ref is None:
            p_cam_ref, p_rtde_ref = p_cam, p_rtde
            
        # Método A: Norma de desplazamiento 3D [mm]
        disp_cam_mm = np.linalg.norm(p_cam - p_cam_ref) * 1000.0
        disp_rtde_mm = np.linalg.norm(p_rtde - p_rtde_ref) * 1000.0
        
        # Método B: Posición de la brida en marco de la Base {0} [m]
        p_base_aruco = p_rtde_ref + R_cam_to_base @ (p_cam - p_cam_ref)
        
        print(f"--- Resultados para {cfg_name} ---")
        print(f" RTDE TCP [x,y,z] (Base): {np.round(p_rtde, 4)} m")
        print(f" ArUco Flange (Cam Frame): {np.round(p_cam, 4)} m")
        print(f" ArUco Flange (Base Frame): {np.round(p_base_aruco, 4)} m")
        print(f" Desplazamiento ||dp_robot||: {disp_rtde_mm:.2f} mm")
        print(f" Desplazamiento ||dp_aruco||: {disp_cam_mm:.2f} mm (Diferencia = {abs(disp_rtde_mm - disp_cam_mm):.2f} mm)")

finally:
    if 'rtde_r' in locals() and rtde_r is not None:
        rtde_r.disconnect()
    if 'cap' in locals() and cap is not None:
        cap.release()
    print("\n[OK] Conexión RTDE cerrada y cámara liberada correctamente.")