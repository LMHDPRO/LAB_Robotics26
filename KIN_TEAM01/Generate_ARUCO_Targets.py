import os
import cv2
import numpy as np

# Obtener la ruta del directorio donde está este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# 1. Generar Marcador ArUco (DICT_4X4_50, ID = 0)
# ---------------------------------------------------------
try:
    # Para OpenCV 4.7.0 o superior
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(aruco_dict, id=0, sidePixels=600)
except AttributeError:
    # Para versiones antiguas de OpenCV (< 4.7.0)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.drawMarker(aruco_dict, 0, 600)

# Borde blanco
marker_padded = cv2.copyMakeBorder(marker, 120, 120, 120, 120, cv2.BORDER_CONSTANT, value=255)

path_aruco = os.path.join(SCRIPT_DIR, "Printable_ArUco_ID0_50mm.png")
success_aruco = cv2.imwrite(path_aruco, marker_padded)

# ---------------------------------------------------------
# 2. Generar Tablero de Ajedrez (8x10 cuadros)
# ---------------------------------------------------------
sq_px = 150
board = np.zeros((8 * sq_px, 10 * sq_px), dtype=np.uint8)

for r in range(8):
    for c in range(10):
        if (r + c) % 2 == 1:
            board[r * sq_px:(r + 1) * sq_px, c * sq_px:(c + 1) * sq_px] = 255

board_padded = cv2.copyMakeBorder(board, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)

path_board = os.path.join(SCRIPT_DIR, "Printable_Checkerboard_7x9_25mm.png")
success_board = cv2.imwrite(path_board, board_padded)

# ---------------------------------------------------------
# Confirmación en consola
# ---------------------------------------------------------
if success_aruco and success_board:
    print("✅ Las imágenes se generaron correctamente en:")
    print(f" 1. {path_aruco}")
    print(f" 2. {path_board}")
else:
    print("Ocurrió un problema al guardar las imágenes.")