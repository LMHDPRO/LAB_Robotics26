import numpy as np

# Parámetros DH Clásicos de Spong para el manipulador UR3e (Sección 5.2 del Manual)
def compute_jacobian(q):
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
        
    o_flange = T[6][:3, 3]  # Origen del marco de la brida {6}
    
    J = np.zeros((6, 6))
    for i in range(6):
        z_prev = T[i][:3, 2]
        o_prev = T[i][:3, 3]
        J[:3, i] = np.cross(z_prev, o_flange - o_prev)
        J[3:, i] = z_prev
    return J

def analyze_workspace():
    # Las 15 configuraciones oficiales de la Tabla 7.4 (Fase D del Manual)
    configurations = [
        ("C01_Home",           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("C02_Elbow15",        [0.0, -1.309, 0.262, -1.571, 0.524, 0.0]),
        ("C03_Elbow30",        [0.0, -1.047, 0.524, -1.571, 0.524, 0.0]),
        ("C04_Elbow45",        [0.0, -0.785, 0.785, -1.571, 0.524, 0.0]),
        ("C05_Elbow60",        [0.0, -0.524, 1.047, -1.571, 0.524, 0.0]),
        ("C06_Rot45",          [0.785, -1.047, 0.524, -1.571, 0.524, 0.0]),
        ("C07_Rot90",          [1.571, -1.047, 0.524, -1.571, 0.524, 0.0]),
        ("C08_WristPi4",       [0.0, -0.785, 0.785, -0.785, 0.785, 0.0]),
        ("C09_WristPi2",       [0.0, -0.785, 0.785, 0.0, 1.571, 0.0]),
        ("C10_Extended",       [0.0, -1.571, 0.0, -1.571, 0.524, 0.0]),
        ("C11_HighElbow",      [0.0, -0.262, 1.309, -1.571, 0.524, 0.0]),
        ("C12_DexterousSafe",  [0.0, -1.571, 1.047, -2.618, 0.524, 0.0]),
        ("C13_MixedAsym",      [-0.524, -1.047, 0.524, -1.047, 0.524, 0.0]),
        ("C14_NearWristSing",  [0.0, -1.571, 1.047, -1.571, 0.087, 0.0]),
        ("C15_BoundaryReach",  [1.571, -1.571, 1.571, -1.571, 1.571, 0.0])
    ]
    
    results = []
    
    for label, q in configurations:
        J = compute_jacobian(q)
        U, S, Vt = np.linalg.svd(J)
        
        # Manipulabilidad de Yoshikawa mu = prod(sigma_i)
        mu = np.prod(S)
        
        # Número de condición kappa = sigma_max / sigma_min
        kappa = S[0] / S[-1] if S[-1] > 1e-12 else S[0] / 1e-18
        
        results.append({
            'label': label,
            'mu': mu,
            'kappa': kappa
        })
        
    # Ordenar de mayor a menor manipulabilidad (Destreza global)
    results.sort(key=lambda x: x['mu'], reverse=True)
    
    # Imprimir tabla con formato idéntico a la consola
    print("=" * 85)
    print(f"{'Rank':<6} | {'Configuration Label':<22} | {'Mu (μ)':<16} | {'Kappa (κ)':<22}")
    print("=" * 85)
    
    for rank, item in enumerate(results, start=1):
        label = item['label']
        mu_str = f"{item['mu']:.6f}"
        
        if item['kappa'] > 1e6:
            kappa_str = f"{item['kappa']:.2f}"
        else:
            kappa_str = f"{item['kappa']:.2f}"
            
        print(f"{rank:<6} | {label:<22} | {mu_str:<16} | {kappa_str:<22}")
        
    print("=" * 85)
    print(f"\nConfiguración con mayor manipulabilidad (Global Max): {results[0]['label']}")

if __name__ == '__main__':
    analyze_workspace()