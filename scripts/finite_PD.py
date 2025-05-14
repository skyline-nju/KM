import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def read_op_series(L, D_psi, J, D_theta=0, sigma=0, rho0=1, h=0.02, v0=1., seed=1002, ret_theta=False):
    pat = f"../data/finite_PD/L{L:d}/op/L{L:d}_{L:d}_r{rho0:g}_v{v0:g}_T{D_psi:g}_J{J:g}_s{sigma:g}_D{D_theta:.4f}_h{h:g}_S{seed:d}_t*.dat"
    files = glob.glob(pat)
    # print(files)
    lines_dict = {}
    n_lines = 0
    for file in files:
        t_beg = int((os.path.basename(file).rstrip(".dat").split("_")[-1]).lstrip("t"))
        with open(file, "r") as fin:
            lines = fin.readlines()
            lines_dict[t_beg] = lines
            n_lines += len(lines)
    phi_arr = np.zeros(n_lines)
    print(n_lines)
    if ret_theta:
        theta_arr = np.zeros_like(phi_arr)
    i = 0
    flag_remove_last_line = False
    for t_beg in sorted(lines_dict.keys()):
        # print(t_beg)
        # print(file)
        lines = lines_dict[t_beg]
        for line in lines:
            s = line.rstrip("\n").split("\t")
            try:
                phi_arr[i] = float(s[1])
                if ret_theta:
                    theta_arr[i] = float(s[2])
            except IndexError:
                phi_arr[i] = phi_arr[i-1]
                if ret_theta:
                    theta_arr[i] = theta_arr[i-1]
            except ValueError:
                phi_arr[i] = phi_arr[i-1]
                if ret_theta:
                    theta_arr[i] = theta_arr[i-1]

            i += 1
    t_arr = (np.arange(phi_arr.size) + 1) * 100 * h
    if not ret_theta:
        return t_arr, phi_arr
    else:
        return t_arr, phi_arr, theta_arr


def PD_J_T(L, ncut=10000):
    J_arr = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
    T_arr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    phi_arr = np.zeros((T_arr.size, J_arr.size))

    for j, T in enumerate(T_arr):
        for i, J in enumerate(J_arr):
            t, phi = read_op_series(L, T, J)
            phi_arr[j, i] = np.mean(phi[ncut:])

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    extent = [J_arr[0]-0.125, J_arr[-1]+0.125, T_arr[0]-0.05, T_arr[-1]+0.05]
    im = ax.imshow(phi_arr, origin="lower", extent=extent, aspect="auto")
    ax.set_xlabel(r"$J$", fontsize="x-large")
    ax.set_ylabel(r"$T$", fontsize="x-large")
    cb = fig.colorbar(im)
    cb.set_label(r"Global Polarity", fontsize="x-large")
    fig.suptitle(r"$L=128, D_\psi = 0, \sigma=0$", fontsize="x-large")
    plt.show()
    plt.close()

if __name__ == "__main__":
    L = 128
    T = 0.3
    J = 1.25

    t_arr, phi_arr, theta_arr = read_op_series(L, T, J, ret_theta=True)

    plt.plot(t_arr, phi_arr)
    plt.show()
    plt.close()
    PD_J_T(L=128)