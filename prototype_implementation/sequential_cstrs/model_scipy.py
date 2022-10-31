from scipy.optimize import fsolve, root


if __name__ == '__main__':
    def f(x, inputs):
        C = x[0:8].reshape((4, 2))
        Tk = x[8:10]
                
        A0 = inputs[0]
        A0_D0 = inputs[1]
        q0 = inputs[2]
        V1 = inputs[3]
        V2 = inputs[4]
        T0 = inputs[5]
        
        k0 = np.array([
            2800,
            12,
        ])
        ea = np.array([
            2995,
            4427,
        ])
        k = k0[:, None] * np.exp(ea[:, None] / (Tk[None, :] + 273))
        
        CAk = C[0, :]
        CDk = C[3, :]
        k1k = k[0]
        R1k = k1k * CAk * CDk
        CBk = C[1, :]
        k2k = k[1]
        R2k = k2k * CBk
        R = np.array([R1k, R2k])
        
        C_in = np.array([
            A0,
            0.000,
            0.000,
            A0 / A0_D0,
        ])
        nu = np.array([
            [-1, 0],
            [1, -1],
            [0, 1],
            [-1, 0],
        ])
        V = np.array([V1, V2])
        material_balance = np.array([
            [q0 * (C_in[species] if reactor == 0 else C[species, reactor - 1]) for reactor in range(2)] for species in range(4)
        ])
        material_balance -= q0 * C
        material_balance += np.sum(nu[:, :, None] * R[None, :, :] * V[None, None, :], axis=1)
        
        rho = 0.80
        cp = 1.7
        delta_Hj = np.array([
            -80,
            0,
        ])
        energy_balance = np.array([
            q0 * rho * cp * (T0 if reactor == 0 else Tk[reactor - 1]) for reactor in range(2)
        ])
        energy_balance -= q0 * rho * cp * Tk + V * np.sum(R * delta_Hj[:, None], axis=0)
        return np.append(100 * material_balance.flatten(), 100 * energy_balance.flatten())


if __name__ == '__main__':
    import numpy as np
    inputs = np.array([
        0.800,
        0.800,
        0.025,
        0.729,
        2.000,
        22.0
    ])
    x0 = np.zeros(10)
    x0[8:10] = 22.0
    out = f(x=x0, inputs=inputs)
    print(out)
    res, infodict, ier, msg = fsolve(
        f,
        x0=x0,
        args=inputs,
        full_output=True,
    )
    print(res)
    print(infodict)
    print(ier)
    print(msg)
    res = root(
        f,
        x0=x0,
        args=inputs,
    )
    print(res)
