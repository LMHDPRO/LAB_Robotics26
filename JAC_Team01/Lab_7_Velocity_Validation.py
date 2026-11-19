import time
import numpy as np
import rtde_control
import rtde_receive

ROBOT_IP = '10.110.20.4'
TCP_Z_OFFSET = 0.1085 # 108.5 mm en metros

def compute_jacobian(q, z_tcp=TCP_Z_OFFSET):
    a = [0, -0.24355, -0.21320, 0, 0, 0]
    alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
    d = [0.15185, 0, 0, 0.13105, 0.08535, 0.09210]
    
    T = [np.eye(4)]
    for i in range(6):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        A = np.array([[ct, -st*ca, st*sa, a[i]*ct],
                      [st, ct*ca, -ct*sa, a[i]*st],
                      [0, sa, ca, d[i]],
                      [0, 0, 0, 1]])
        T.append(T[-1] @ A)
        
    o_flange = T[6][:3, 3]
    z6 = T[6][:3, 2]
    o_tcp = o_flange + z_tcp * z6
    
    J = np.zeros((6, 6))
    for i in range(6):
        z_prev = T[i][:3, 2]
        o_prev = T[i][:3, 3]
        J[:3, i] = np.cross(z_prev, o_tcp - o_prev)
        J[3:, i] = z_prev
    return J

def test_velocity(rtde_c, rtde_r, q_target, qdot_cmd, row_idx):
    # 1. Mover a la postura objetivo de la prueba
    rtde_c.moveJ(q_target, speed=0.4, acceleration=0.3)
    time.sleep(0.4)
    
    # 2. Iniciar velocidad continua (time=0.0)
    rtde_c.speedJ(qdot_cmd, acceleration=0.5, time=0.0)
    time.sleep(0.8) # Muestreo en velocidad estable
    
    # 3. Muestreo de sensores reales
    q_act = rtde_r.getActualQ()
    qdot_act = rtde_r.getActualQd()
    v_meas = np.array(rtde_r.getActualTCPSpeed()[:3])
    
    # 4. Detener el comando de velocidad de forma limpia sin matar el hilo
    rtde_c.speedStop(2.0)
    time.sleep(0.5) # Pausa para que las articulaciones desaceleren
    
    # 5. Cálculo de velocidad predicha con el Jacobeano
    J = compute_jacobian(q_act, z_tcp=TCP_Z_OFFSET)
    v_pred = J[:3, :] @ np.array(qdot_act)
    
    # 6. Métrica de error en mm/s
    delta_v = np.linalg.norm(v_pred - v_meas) * 1000.0
    passed = "PASS" if delta_v < 5.0 else "FAIL"
    
    v_pred_mm = np.round(v_pred * 1000, 1)
    v_meas_mm = np.round(v_meas * 1000, 1)
    
    print(f"Fila {row_idx:02d} | v_pred: {v_pred_mm} | v_RTDE: {v_meas_mm} | Δv: {delta_v:.2f} mm/s | [{passed}]")
    return delta_v

if __name__ == '__main__':
    print("Conectando con el UR3e...")
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    time.sleep(1.0)

    # Las 5 Configuraciones del Manual (C1 - C5)
    configs = [
        [0.0, -np.pi/2, 0.0, -np.pi/2, np.pi/6, 0.0],              # C1
        [np.pi/4, -np.pi/3, np.pi/6, -np.pi/2, np.pi/4, 0.0],       # C2
        [np.pi/2, -np.pi/4, np.pi/4, -3*np.pi/4, np.pi/3, 0.0],     # C3
        [-np.pi/4, -np.pi/2, np.pi/3, -5*np.pi/6, np.pi/6, 0.0],    # C4
        [0.0, -np.pi/3, np.pi/2, -np.pi/2, np.pi/4, 0.0]            # C5
    ]

    # Vectores de entrada qdot (0.1 rad/s en articulaciones 1, 2 y 3)
    inputs = [
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
    ]

    q_targets = []
    qdot_cmds = []
    for cfg in configs:
        for qd in inputs:
            q_targets.append(cfg)
            qdot_cmds.append(qd)

    errors = []
    print("\n--- INICIANDO FASE A: VALIDACIÓN DE 15 FILAS CON TCP = 108.5 mm ---")
    try:
        for i in range(15):
            err = test_velocity(rtde_c, rtde_r, q_targets[i], qdot_cmds[i], i + 1)
            errors.append(err)
            time.sleep(0.3)

        print("\n================ RESUMEN TABLA 7.1 ================")
        print(f"Error promedio: {np.mean(errors):.2f} mm/s (Tolerancia manual: < 5.0 mm/s)")
        print(f"Error máximo: {np.max(errors):.2f} mm/s | Error mínimo: {np.min(errors):.2f} mm/s")

    except KeyboardInterrupt:
        print("\nPrueba cancelada por el usuario.")
    finally:
        rtde_c.speedStop(2.0)
        rtde_c.stopScript()