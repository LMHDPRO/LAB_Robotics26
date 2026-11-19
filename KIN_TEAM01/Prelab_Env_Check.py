import sys, cv2, numpy as np, rtde_receive

print("Python Version:", sys.version.split()[0])
print("OpenCV Version:", cv2.__version__)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

print("[OK] ur_rtde y OpenCV ArucoDetector inicializados correctamente!")