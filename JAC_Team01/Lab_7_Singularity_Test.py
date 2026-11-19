import time
import numpy as np
import rtde_control
import rtde_receive

ROBOT_IP = '10.110.20.4'
TCP_Z_OFFSET = 0.1085  # 108.5 mm en metros
LAMBDA_DLS = 0.01       # Factor de amortiguamiento DLS según manual

def compute_jacobian(q, z_tcp=TCP_Z_OFFSET):
    # Parámetros Spong DH del UR3e
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

def analyze_posture(rtde_c, rtde_r, q_target, posture_name, x_dot_des):
    # 1. Posicionar el robot de forma segura
    rtde_c.moveJ(q_target, speed=0.4, acceleration=0.3)
    time.sleep(0.5)
    
    # 2. Leer encoders reales
    q_act = rtde_r.getActualQ()
    J = compute_jacobian(q_act, z_tcp=TCP_Z_OFFSET)
    
    # 3. Indicadores de Singularidad mediante SVD
    U, S, Vt = np.linalg.svd(J)
    det_J = np.abs(np.linalg.det(J))
    sigma_max = S[0]
    sigma_min = S[-1]
    kappa_J = sigma_max / sigma_min if sigma_min > 1e-9 else np.inf
    mu_J = np.prod(S)  # Manipulabilidad de Yoshikawa
    
    # 4. Inversión Cinemática: Pseudoinversa Standard (pinv)
    J_pinv = np.linalg.pinv(J)
    qdot_pinv = J_pinv @ x_dot_des
    norm_pinv = np.linalg.norm(qdot_pinv)
    
    # 5. Inversión Cinemática: Damped Least-Squares (DLS)
    J_JT = J @ J.T
    dls_inv = np.linalg.inv(J_JT + (LAMBDA_DLS**2) * np.eye(6))
    qdot_dls = J.T @ dls_inv @ x_dot_des
    norm_dls = np.linalg.norm(qdot_dls)
    
    # 6. Relación DLS / pinv
    ratio = norm_dls / norm_pinv if norm_pinv > 1e-9 else 0.0
    
    # Formatear salida para consola
    print(f"{posture_name:<25} | {det_J:10.6f} | {kappa_J:10.2f} | {mu_J:10.6f} | {norm_pinv:10.4f} | {norm_dls:10.4f} | {ratio:10.4f}")
    
    return {
        'name': posture_name,
        'det': det_J,
        'kappa': kappa_J,
        'mu': mu_J,
        'pinv': norm_pinv,
        'dls': norm_dls,
        'ratio': ratio
    }

if __name__ == '__main__':
    print("Conectando con el UR3e...")
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    time.sleep(1.0)

    # Comando cartesiano deseado: 50 mm/s en el eje X (0.05 m/s)
    x_dot_des = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Definición de las 6 posturas de la Tabla 7.2 del manual
    postures = [
        # Barrido de Codo (Joint 3)
        ("Elbow Safe (q3=60°)",     [0.0, -np.pi/2, np.deg2rad(60), -np.pi/2, np.deg2rad(30), 0.0]),
        ("Elbow Approach (q3=30°)", [0.0, -np.pi/2, np.deg2rad(30), -np.pi/2, np.deg2rad(30), 0.0]),
        ("Elbow Near-Sing (q3=10°)",[0.0, -np.pi/2, np.deg2rad(10), -np.pi/2, np.deg2rad(30), 0.0]),
        
        # Barrido de Muñeca (Joint 5)
        ("Wrist Safe (q5=30°)",     [0.0, -np.pi/2, np.deg2rad(60), -np.pi/2, np.deg2rad(30), 0.0]),
        ("Wrist Near-Sing (q5=5°)", [0.0, -np.pi/2, np.deg2rad(60), -np.pi/2, np.deg2rad(5),  0.0]),
        ("Wrist Exact (q5=0°)",     [0.0, -np.pi/2, np.deg2rad(60), -np.pi/2, np.deg2rad(0),  0.0]),
    ]

    print("\n" + "="*105)
    print("--- INICIANDO FASE B: BARRIDO DE SINGULARIDADES Y REGULARIZACIÓN DLS (TABLA 7.2) ---")
    print("="*105)
    print(f"{'Test Posture':<25} | {'|det(J)|':<10} | {'kappa(J)':<10} | {'mu(J)':<10} | {'||qd|| pinv':<10} | {'||qd|| DLS':<10} | {'Ratio DLS/pinv':<10}")
    print("-" * 105)

    results = []
    try:
        for name, q_target in postures:
            res = analyze_posture(rtde_c, rtde_r, q_target, name, x_dot_des)
            results.append(res)
            time.sleep(0.3)

        print("="*105)
        print("Prueba finalizada con éxito. Copia los valores impresos a la Tabla 7.2 de tu reporte.")

    except KeyboardInterrupt:
        print("\nPrueba cancelada por el usuario.")
    finally:
        rtde_c.stopScript()