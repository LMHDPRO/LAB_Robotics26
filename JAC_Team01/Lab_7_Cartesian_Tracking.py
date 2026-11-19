import sys
import time
import numpy as np
import rtde_control
import rtde_receive

# Detector no bloqueante de tecla 'X' para Windows y Linux/Mac
if sys.platform == "win32":
    import msvcrt
    def check_keyboard_stop():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch.lower() in [b'x', b'\x03']:
                return True
        return False
else:
    import select
    def check_keyboard_stop():
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch.lower() in ['x', '\x03']:
                return True
        return False

# Hardware y Configuración de Laboratorio UR3e
ROBOT_IP = '10.110.20.4'
TCP_Z_OFFSET = 0.1085     # 108.5 mm (Offset de herramienta)
PAYLOAD_KG = 0.550        # Payload del gripper (0.550 kg)
PENDANT_SCALE = 0.13      # Regla V1: Ajuste del 13% en Teach Pendant

MAX_JOINT_SPEED_CMD = 2.0 # Límite de comando en Python (Físico real <= 0.26 rad/s)
KP_POS = 1.8              # Ganancia proporcional calibrada

def compute_jacobian_and_fk(q, z_tcp=TCP_Z_OFFSET):
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
    return J, o_tcp

def apply_vector_scaling(qdot_cmd, max_limit=MAX_JOINT_SPEED_CMD):
    """Escala el vector proporcionalmente para mantener la dirección cartesiana."""
    max_val = np.max(np.abs(qdot_cmd))
    if max_val > max_limit:
        return qdot_cmd * (max_limit / max_val)
    return qdot_cmd

def run_straight_line(rtde_c, rtde_r, distance_m=0.100, nominal_speed_mps=0.050):
    print("\n--- Ejecutando Trayectoria 1: Línea Recta (+X, 100 mm) ---")
    print(" Presiona la tecla 'X' para detener en cualquier momento.")
    dt = 0.02
    
    eff_speed_mps = nominal_speed_mps * PENDANT_SCALE
    total_time = distance_m / eff_speed_mps
    steps = int(total_time / dt)
    
    q_act = rtde_r.getActualQ()
    _, p_start = compute_jacobian_and_fk(q_act)
    p_expected = p_start + np.array([distance_m, 0.0, 0.0])
    
    max_qdot_act_observed = 0.0
    min_mu_observed = np.inf
    
    # 1. Fase de trayectoria dinámica
    for i in range(steps):
        if check_keyboard_stop():
            raise KeyboardInterrupt("Detención manual solicitada por el usuario (Tecla X).")
            
        t_start = time.time()
        q_act = rtde_r.getActualQ()
        J, p_act = compute_jacobian_and_fk(q_act)
        
        S = np.linalg.svd(J, compute_uv=False)
        mu = np.prod(S)
        min_mu_observed = min(min_mu_observed, mu)
        
        p_des = p_start + np.array([eff_speed_mps * (i * dt), 0.0, 0.0])
        v_phys_des = np.array([eff_speed_mps, 0.0, 0.0]) + KP_POS * (p_des - p_act)
        
        x_dot_cmd = np.zeros(6)
        x_dot_cmd[:3] = v_phys_des / PENDANT_SCALE
        
        qdot_cmd = np.linalg.pinv(J) @ x_dot_cmd
        qdot_cmd = apply_vector_scaling(qdot_cmd, MAX_JOINT_SPEED_CMD)
        
        qdot_act_real = qdot_cmd * PENDANT_SCALE
        max_qdot_act_observed = max(max_qdot_act_observed, np.max(np.abs(qdot_act_real)))
        
        rtde_c.speedJ(qdot_cmd.tolist(), acceleration=1.5, time=0.0)
        
        elapsed = time.time() - t_start
        if dt > elapsed:
            time.sleep(dt - elapsed)
            
    # 2. Fase de Asentamiento (Settling phase para asegurar convergencia < 1.0 mm)
    settle_timeout = 2.0  # Máximo 2 segundos de ajuste fino
    t_settle_start = time.time()
    while time.time() - t_settle_start < settle_timeout:
        if check_keyboard_stop():
            raise KeyboardInterrupt("Detención manual por el usuario (Tecla X).")
            
        t_start = time.time()
        q_act = rtde_r.getActualQ()
        J, p_act = compute_jacobian_and_fk(q_act)
        
        err = p_expected - p_act
        if np.linalg.norm(err) < 0.001:  # Menor a 1.0 mm
            break
            
        v_phys_des = KP_POS * err
        x_dot_cmd = np.zeros(6)
        x_dot_cmd[:3] = v_phys_des / PENDANT_SCALE
        
        qdot_cmd = np.linalg.pinv(J) @ x_dot_cmd
        qdot_cmd = apply_vector_scaling(qdot_cmd, MAX_JOINT_SPEED_CMD)
        
        rtde_c.speedJ(qdot_cmd.tolist(), acceleration=1.5, time=0.0)
        
        elapsed = time.time() - t_start
        if dt > elapsed:
            time.sleep(dt - elapsed)
            
    rtde_c.speedStop(2.0)
    time.sleep(0.5)
    
    q_final = rtde_r.getActualQ()
    _, p_final = compute_jacobian_and_fk(q_final)
    delta_p = np.linalg.norm(p_expected - p_final) * 1000.0  # Error en mm
    
    passed = "PASS" if delta_p < 8.0 else "FAIL"
    print(f"Resultado Línea Recta: Error delta_p = {delta_p:.2f} mm | Max |qd_real| = {max_qdot_act_observed:.3f} rad/s | Min mu = {min_mu_observed:.6f} | [{passed}]")
    return delta_p, max_qdot_act_observed, min_mu_observed, passed

def run_circular_arc(rtde_c, rtde_r, radius_m=0.050, nominal_v_tan_mps=0.030):
    print("\n--- Ejecutando Trayectoria 2: Arco Circular (XY, r=50 mm) ---")
    print(" Presiona la tecla 'X' para detener en cualquier momento.")
    dt = 0.02
    
    eff_v_tan_mps = nominal_v_tan_mps * PENDANT_SCALE
    w_radps = eff_v_tan_mps / radius_m
    total_time = (2.0 * np.pi) / w_radps
    steps = int(total_time / dt)
    
    q_act = rtde_r.getActualQ()
    _, p_start = compute_jacobian_and_fk(q_act)
    p_center = p_start + np.array([radius_m, 0.0, 0.0])
    
    max_qdot_act_observed = 0.0
    min_mu_observed = np.inf
    
    # 1. Fase de arco circular dinámico
    for i in range(steps):
        if check_keyboard_stop():
            raise KeyboardInterrupt("Detención manual por el usuario (Tecla X).")
            
        t_start = time.time()
        t = i * dt
        q_act = rtde_r.getActualQ()
        J, p_act = compute_jacobian_and_fk(q_act)
        
        S = np.linalg.svd(J, compute_uv=False)
        mu = np.prod(S)
        min_mu_observed = min(min_mu_observed, mu)
        
        p_des = p_center + np.array([-radius_m * np.cos(w_radps * t), radius_m * np.sin(w_radps * t), 0.0])
        v_phys_des = np.array([eff_v_tan_mps * np.sin(w_radps * t), eff_v_tan_mps * np.cos(w_radps * t), 0.0]) + KP_POS * (p_des - p_act)
        
        x_dot_cmd = np.zeros(6)
        x_dot_cmd[:3] = v_phys_des / PENDANT_SCALE
        
        qdot_cmd = np.linalg.pinv(J) @ x_dot_cmd
        qdot_cmd = apply_vector_scaling(qdot_cmd, MAX_JOINT_SPEED_CMD)
        
        qdot_act_real = qdot_cmd * PENDANT_SCALE
        max_qdot_act_observed = max(max_qdot_act_observed, np.max(np.abs(qdot_act_real)))
        
        rtde_c.speedJ(qdot_cmd.tolist(), acceleration=1.5, time=0.0)
        
        elapsed = time.time() - t_start
        if dt > elapsed:
            time.sleep(dt - elapsed)
            
    # 2. Fase de Asentamiento de cierre de círculo
    settle_timeout = 2.0
    t_settle_start = time.time()
    while time.time() - t_settle_start < settle_timeout:
        if check_keyboard_stop():
            raise KeyboardInterrupt("Detención manual por el usuario (Tecla X).")
            
        t_start = time.time()
        q_act = rtde_r.getActualQ()
        J, p_act = compute_jacobian_and_fk(q_act)
        
        err = p_start - p_act
        if np.linalg.norm(err) < 0.001:
            break
            
        v_phys_des = KP_POS * err
        x_dot_cmd = np.zeros(6)
        x_dot_cmd[:3] = v_phys_des / PENDANT_SCALE
        
        qdot_cmd = np.linalg.pinv(J) @ x_dot_cmd
        qdot_cmd = apply_vector_scaling(qdot_cmd, MAX_JOINT_SPEED_CMD)
        
        rtde_c.speedJ(qdot_cmd.tolist(), acceleration=1.5, time=0.0)
        
        elapsed = time.time() - t_start
        if dt > elapsed:
            time.sleep(dt - elapsed)
            
    rtde_c.speedStop(2.0)
    time.sleep(0.5)
    
    q_final = rtde_r.getActualQ()
    _, p_end = compute_jacobian_and_fk(q_final)
    delta_circ = np.linalg.norm(p_start - p_end) * 1000.0  # Error de cierre de círculo en mm
    
    passed = "PASS" if delta_circ < 10.0 else "FAIL"
    print(f"Resultado Círculo: Error delta_circ = {delta_circ:.2f} mm | Max |qd_real| = {max_qdot_act_observed:.3f} rad/s | Min mu = {min_mu_observed:.6f} | [{passed}]")
    return delta_circ, max_qdot_act_observed, min_mu_observed, passed

if __name__ == '__main__':
    print("Conectando con el robot UR3e...")
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    time.sleep(1.0)
    
    # Configuración de TCP y Payload en PolyScope a través de RTDE
    rtde_c.setTcp([0.0, 0.0, TCP_Z_OFFSET, 0.0, 0.0, 0.0])
    rtde_c.setPayload(PAYLOAD_KG, [0.0, 0.0, TCP_Z_OFFSET / 2.0])
    time.sleep(0.5)
    
    q_safe = [0.0, -np.pi/2, np.pi/3, -5*np.pi/6, np.pi/6, 0.0]
    
    try:
        # 1. Ejecución de Línea Recta
        print("Llevando robot a postura inicial segura (q_safe)...")
        rtde_c.moveJ(q_safe, speed=0.4, acceleration=0.3)
        time.sleep(1.0)
        line_err, line_qd, line_mu, line_pass = run_straight_line(rtde_c, rtde_r)
        
        # 2. Retorno obligatorio a q_safe antes del Círculo
        print("\nRegresando a postura inicial segura (q_safe) para despejar el espacio de trabajo...")
        rtde_c.moveJ(q_safe, speed=0.4, acceleration=0.3)
        time.sleep(1.0)
        circ_err, circ_qd, circ_mu, circ_pass = run_circular_arc(rtde_c, rtde_r)
        
        # 3. Resumen final para la Tabla 7.3
        print("\n================ RESUMEN TABLA 7.3 ================")
        print(f"Línea Recta (+X) | Error: {line_err:.2f} mm (< 8.0 mm) | Max qd real: {line_qd:.3f} rad/s | Min mu: {line_mu:.6f} | [{line_pass}]")
        print(f"Arco Circular (XY)| Error: {circ_err:.2f} mm (< 10.0 mm) | Max qd real: {circ_qd:.3f} rad/s | Min mu: {circ_mu:.6f} | [{circ_pass}]")

    except (KeyboardInterrupt, Exception) as e:
        print(f"\n[PARADA DE SEGURIDAD / EXCEPCIÓN]: {e}")
    finally:
        print("Finalizando conexión y liberando RTDE...")
        try:
            if rtde_c.isConnected():
                rtde_c.speedStop(2.0)
                rtde_c.stopScript()
        except Exception:
            pass
        print("Programa terminado de forma limpia.")