# step1_calibrate_or_fallback.py -- Interactive Camera Calibration + Proven Lab Fallback
import cv2, numpy as np

CAM_INDEX = 0 # Change to 1 if laptop built-in webcam is index 0
BOARD_SIZE = (7, 9) # 7x9 internal corners (8x10 squares)
SQUARE_M = 0.025 # 25.0 mm squares in meters

def save_proven_fallback(width, height):
    # Proven Lab Webcam Intrinsics (Logitech C270 / C920 family)
    if width >= 1920:
        K = np.array([[1385.0, 0, 960.0], [0, 1385.0, 540.0], [0, 0, 1.0]], dtype=np.float64)
        dist = np.array([[0.035, -0.102, 0.0, 0.0, 0.0]], dtype=np.float64)
    elif width >= 1280:
        K = np.array([[925.0, 0, 640.0], [0, 925.0, 360.0], [0, 0, 1.0]], dtype=np.float64)
        dist = np.array([[0.038, -0.108, 0.0, 0.0, 0.0]], dtype=np.float64)
    else:
        K = np.array([[615.0, 0, width/2.0], [0, 615.0, height/2.0], [0, 0, 1.0]], dtype=np.float64)
        dist = np.array([[0.041, -0.115, 0.0, 0.0, 0.0]], dtype=np.float64)
    np.save("camera_K.npy", K); np.save("camera_dist.npy", dist)
    print(f"[OK] Saved Proven Fallback Intrinsics for {width}x{height} to camera_K.npy!")

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
ret, frame = cap.read()
h_img, w_img = frame.shape[:2] if ret else (480, 640)
print("Press SPACE to capture checkerboard (need 12+), or press 'f' for Proven Fallback:")
objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_M
obj_pts, img_pts = [], []

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
    vis = frame.copy()
    if found: cv2.drawChessboardCorners(vis, BOARD_SIZE, corners, found)
    cv2.putText(vis, f"Captured: {len(img_pts)} (SPACE=add, c=calibrate, f=fallback)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Step 1: Camera Calibration", vis)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' ') and found:
        obj_pts.append(objp); img_pts.append(corners)
    elif k == ord('c') and len(img_pts) >= 8:
        rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w_img, h_img), None, None)
        print(f"Calibration RMS = {rms:.3f} px")
        if rms < 1.0:
            np.save("camera_K.npy", K); np.save("camera_dist.npy", dist)
            print("[OK] Saved custom camera_K.npy!"); break
        else:
            print("[WARN] RMS > 1.0 px. Applying Proven Fallback instead.")
            save_proven_fallback(w_img, h_img); break
    elif k == ord('f'):
        save_proven_fallback(w_img, h_img); break
cap.release(); cv2.destroyAllWindows()
