import rtde_receive

ROBOT_IP = "169.254.15.34"

puntos_guardados = []

print("=== Captura de Posiciones ===")
print("Instrucciones: Mueve el robot, escribe 's' y presiona Enter para guardar.\n")

while len(puntos_guardados) < 5:
    tecla = input(f"[{len(puntos_guardados) + 1}/5] Ingresa 's' para guardar posición actual: ").strip().lower()
    
    if tecla == 's':
        try:
            # Se conecta en vivo únicamente para leer las coordenadas
            rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
            
            q = rtde_r.getActualQ()
            pose = rtde_r.getActualTCPPose()
            
            rtde_r.disconnect()
            
            q_round = [round(v, 4) for v in q]
            pose_round = [round(v, 4) for v in pose]
            
            puntos_guardados.append({
                "q": q_round,
                "pose": pose_round
            })
            
            print(f" -> Guardado correctamente ({len(puntos_guardados)}/5):")
            print("    Joint positions (rad):", q_round)
            print("    TCP pose [x,y,z,Rx,Ry,Rz]:", pose_round)
            print("-" * 50)
            
        except Exception as e:
            print(f" Error al comunicarse con el robot: {e}")
            
    else:
        print("    Tecla no válida. Presiona únicamente la letra 's' y luego Enter.")

print("\n=== Resumen de los 5 puntos guardados ===")
for idx, punto in enumerate(puntos_guardados, start=1):
    print(f"Punto {idx}:")
    print("  Joints (rad):", punto["q"])
    print("  TCP pose:    ", punto["pose"])