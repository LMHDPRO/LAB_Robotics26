# step2_live_aruco_preview.py -- Live Interactive ArUco Detection & Axis Overlay
import cv2, numpy as np

CAM_INDEX = 0
MARKER_ID = 0

MARKER_SIZE = 0.050 # [m] Exact outer black border width (50.0 mm)
K = np.load("camera_K.npy")
dist = np.load("camera_dist.npy")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
h = MARKER_SIZE / 2.0
obj_pts = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
print("Live ArUco Preview running. Press 's' to save debug image, 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and MARKER_ID in ids.flatten():
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        idx = list(ids.flatten()).index(MARKER_ID)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, corners[idx][0], K, dist)
        if ok:
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, h)
            tx, ty, tz = tvec.ravel()
            cv2.putText(frame, f"ID 0 tvec [m]: x={tx:.4f}, y={ty:.4f}, z={tz:.4f}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "MARKER ID 0 NOT DETECTED - Check lighting & flange angle", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.imshow("Step 2: Front-Camera ArUco Live Pose Preview", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("aruco_debug_overlay.png", frame)
        print("[OK] Saved aruco_debug_overlay.png for Lab Report!")
    elif key == ord('q'): break
cap.release(); cv2.destroyAllWindows()
