from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import numpy as np

def model(x, d=None):
    if d is None:
        d = [10, 400]

    R = d[0]
    tau = d[1]

    i_set = [
        "AH",
        "B",
        "A-",
        "BH+",
        "C",
        "AC-",
        "P",
    ]
    j_set = [
        1,
        2,
        3,
        4,
        5,
    ]

    x = np.around(x, decimals=13)

    c = {
        "AH"    : x[0],
        "B"     : x[1],
        "A-"    : x[2],
        "BH+"   : x[3],
        "C"     : x[4],
        "AC-"   : x[5],
        "P"     : x[6],
    }

    nu = {
        ("AH", 1): -1,
        ("AH", 2):  0,
        ("AH", 3):  0,
        ("AH", 4): -1,
        ("AH", 5):  0,

        ("B", 1): -1,
        ("B", 2):  0,
        ("B", 3):  0,
        ("B", 4):  0,
        ("B", 5):  1,

        ("A-", 1):  1,
        ("A-", 2): -1,
        ("A-", 3):  1,
        ("A-", 4):  1,
        ("A-", 5):  0,

        ("BH+", 1):  1,
        ("BH+", 2):  0,
        ("BH+", 3):  0,
        ("BH+", 4):  0,
        ("BH+", 5): -1,

        ("C", 1):  0,
        ("C", 2): -1,
        ("C", 3):  1,
        ("C", 4):  0,
        ("C", 5):  0,

        ("AC-", 1):  0,
        ("AC-", 2):  1,
        ("AC-", 3): -1,
        ("AC-", 4): -1,
        ("AC-", 5): -1,

        ("P", 1):  0,
        ("P", 2):  0,
        ("P", 3):  0,
        ("P", 4):  1,
        ("P", 5):  1,
    }

    k = {
        1: 49.7796,
        2:  8.9316,
        3:  1.3177,
        4:  0.3109,
        5:  3.8781,
    }

    c_in = {
        "AH": 0.3955,
        "B": 0.3955 / R,
        "C": 0.25,
        "BH+": 0,
        "A-": 0,
        "AC-": 0,
        "P": 0,
    }

    r = {
        1: k[1] * c["AH"] * c["B"],
        2: k[2] * c["A-"] * c["C"],
        3: k[3] * c["AC-"],
        4: k[4] * c["AC-"] * c["AH"],
        5: k[5] * c["AC-"] * c["BH+"],
    }

    f = np.array([
        c_in[i] - c[i] + tau * sum(nu[i, j] * r[j] for j in j_set) for i in i_set
    ])

    return f

def jacobian(x, d=None):
    if d is None:
        d = [10, 400]

    R = d[0]
    tau = d[1]

    jac = np.array([

    ])

    return jac

def simulate_c(d, full_output=False):
    R = d[0]
    tau = d[1]

    fsolve_out = fsolve(
        model,
        np.ones(7),
        args=d,
        full_output=full_output,
    )
    if full_output:
        c, infodict, ier, mesg = fsolve_out
        return c, infodict, ier, mesg
    else:
        return fsolve_out

def simulate_g(d, full_output=False):
    R = d[0]
    tau = d[1]

    fsolve_out = fsolve(model, np.ones(7), args=d, full_output=full_output)

    if full_output:
        c, infodict, ier, mesg = fsolve_out
    else:
        c = fsolve_out

    cqa_1 = c[4] + c[5] - 0.1 * 0.25            # conversion of feed C
    cqa_2 = c[5] - 0.002                        # concentration of AC-
    g = np.array([cqa_1, cqa_2]).T

    if full_output:
        return g, infodict, ier, mesg
    else:
        return g

def simulate_cqa(d, full_output=False):
    R = d[0]
    tau = d[1]

    fsolve_out = fsolve(model, np.ones(7), args=d, full_output=full_output)

    if full_output:
        c, infodict, ier, mesg = fsolve_out
    else:
        c = fsolve_out

    cqa = np.array([
        0.25 - c[4] - c[5] / 0.25,
        c[5],
    ])

    if full_output:
        return cqa, infodict, ier, mesg
    else:
        return cqa

def multvar_sim_cqa(exp):
    cqa_list = []
    for one_exp in exp:
        R = one_exp[0]
        tau = one_exp[1]
        cqa, infodict, ier, mesg = simulate_cqa(
            d=[R, tau],
            full_output=True
        )
        if ier != 1:
            print(f"Fsolve failed at R: {R} and tau: {tau}")
            cqa_list.append(np.full_like(cqa, fill_value=np.nan))
        else:
            cqa_list.append(cqa)
    cqa_list = np.array(cqa_list)
    return cqa_list


if __name__ == '__main__':
    d = np.array([10.948678713150818, 1075.7103186259467])
    cqa = simulate_cqa(d)
    print(cqa)
