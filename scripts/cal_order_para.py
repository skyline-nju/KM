import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from add_line import add_line


def read_op_series(L, rho0, D_psi, sigma, D_theta, h=0.1, v0=1., seed=1000, ret_theta=False):
    pat = f"../data/op/L{L:d}_{L:d}_r{rho0:g}_v{v0:g}_T{D_psi:g}_s{sigma:g}_D{D_theta:.4f}_h{h:g}_S{seed:d}_t*.dat"
    files = glob.glob(pat)
    lines_dict = {}
    n_lines = 0
    for file in files:
        t_beg = int((os.path.basename(file).rstrip(".dat").split("_")[-1]).lstrip("t"))
        with open(file, "r") as fin:
            lines = fin.readlines()
            lines_dict[t_beg] = lines
            n_lines += len(lines)
    phi_arr = np.zeros(n_lines)
    if ret_theta:
        theta_arr = np.zeros_like(phi_arr)
    i = 0
    flag_remove_last_line = False
    for t_beg in sorted(lines_dict.keys()):
        print(t_beg)
        print(file)
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


def varied_D_theta():
    rho0 = 1
    D_psi = 0.1
    sigma = 0.2
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4.5, 4.5))

    for D_theta in [0, 0.01, 0.1, 1.]:
        if D_theta == 0.01 or D_theta == 0.:
            L_arr = np.array([32, 64, 128, 256, 512, 1024, 2048])
        elif D_theta == 0.1:
            L_arr = np.array([32, 64, 128, 256, 512, 1024, 2048])
        elif D_theta == 1.:
            L_arr = np.array([32, 64, 128, 256, 512])

        phi_m = np.zeros(L_arr.size)


        for i, L in enumerate(L_arr):
            t_arr, phi_arr = read_op_series(L, rho0, D_psi, sigma, D_theta)

            phi_m[i] = np.mean(phi_arr[10000:])
        ax.plot(L_arr**2, phi_m, "-o", label=r"$%g$" % D_theta)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(title=r"$v_0=1,D_\psi=$", title_fontsize="x-large", fontsize="x-large")

    ax.set_xlabel(r"$N$", fontsize="x-large")
    ax.set_ylabel(r"$P$", fontsize="x-large")
    add_line(ax, 0, 1, 1, -1/16, label=r"$N^{-1/16}$", xl=0.6, yl=0.81)
    add_line(ax, 0, 1, 1, -1/2, label=r"$N^{-1/2}$", yl=0.5)
    plt.show()
    plt.close()


def varied_sigma():
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4.5, 7.5))
    rho0 = 1
    D_psi = 0.1   # Temperature for spins
    D_theta = 0
    for sigma in [0, 0.05, 0.1, 0.2]:
        if sigma == 0:
            L_arr = np.array([32, 64, 128, 256, 512, 1024])
            ncuts = [20000, 20000, 20000, 20000, 40000, 40000]
        elif sigma == 0.01:
            L_arr = np.array([32, 64, 128, 256, 512, 1024])
            ncuts = [20000, 20000, 20000, 20000, 30000, 40000]
        if sigma == 0.05:
            L_arr = np.array([32, 64, 128, 256, 512, 1024])
            ncuts = [20000, 20000, 20000, 20000, 20000, 17500]
        elif sigma == 0.1:
            L_arr = np.array([32, 64, 128, 256, 512, 1024])
            ncuts = [20000, 20000, 20000, 20000, 20000, 20000]
        elif sigma == 0.2:
            L_arr = np.array([32, 64, 128, 256, 512, 1024, 2048])
            ncuts = [20000, 20000, 20000, 20000, 20000, 20000, 30000]
        phi_m = np.zeros(L_arr.size)
        for i, L in enumerate(L_arr):
            t_arr, phi_arr = read_op_series(L, rho0, D_psi, sigma, D_theta)

            print(phi_arr.size)
            phi_m[i] = np.mean(phi_arr[ncuts[i]:])
        ax.plot(L_arr**2, phi_m, "-o", label=r"$%g$" % sigma)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(title=r"$v_0=1,\sigma=$", title_fontsize="x-large", fontsize="x-large")

    ax.set_xlabel(r"$N$", fontsize="x-large")
    ax.set_ylabel(r"$P$", fontsize="x-large")

    add_line(ax, 0, 1, 1, -1/16, label=r"$N^{-1/16}$", xl=0.65, yl=0.85)
    add_line(ax, 0, 0.95, 1, -0.00625)
    add_line(ax, 0, 0.83, 1, -1/16)
    add_line(ax, 0, 1, 1, -1/2, label=r"$N^{-1/2}$", yl=0.5)
    plt.show()
    plt.close()


# varied temperature for spins
def varied_D_psi():
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(4.5, 7.5))
    rho0 = 1
    D_psi = 0.1
    sigma = 0.
    D_theta = 0
    for D_psi in [0.1, 0.2, 0.3, 0.4, 0.5, 0.53, 0.55, 0.6]:
        if D_psi == 0.1:
            L_arr = np.array([32, 64, 128, 256, 512, 1024, 2048])
            ncuts = [20000, 20000, 20000, 20000, 40000, 60000, 120000]
            seeds = [1000, 1000, 1000, 1000, 1000, 1000, 1000]
        elif D_psi == 0.2 or D_psi == 0.3 or D_psi == 0.4:
            L_arr = np.array([32, 64, 128, 256])
            ncuts = [20000, 20000, 20000, 5000]
            seeds = [1000, 1000, 1000, 1000]
        elif D_psi == 0.5:
            L_arr = np.array([32, 64, 128, 256, 512, 1024, 2048])
            ncuts = [20000, 20000, 20000, 20000, 20000, 40000, 120000]
            seeds = [1000, 1000, 1000, 1000, 1000, 1001, 1000]
        elif D_psi == 0.53:
            L_arr = np.array([32, 64, 128, 256, 512])
            ncuts = [20000, 20000, 20000, 20000, 10000]
            seeds = [1000, 1000, 1000, 1000, 1000]
        elif D_psi == 0.55 or D_psi == 0.6:
            L_arr = np.array([32, 64, 128, 256, 512])
            ncuts = [20000, 10000, 10000, 10000, 10000]
            seeds = [1000, 1000, 1000, 1000, 1000]

        phi_m = np.zeros(L_arr.size)


        for i, L in enumerate(L_arr):
            t_arr, phi_arr = read_op_series(L, rho0, D_psi, sigma, D_theta, seed=seeds[i])

            print(phi_arr.size)
            phi_m[i] = np.mean(phi_arr[ncuts[i]:])
        ax.plot(L_arr**2, phi_m, "-o", label=r"$%g$" % D_psi)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.legend(title=r"$v_0=1,T=$", title_fontsize="x-large", fontsize="x-large")

    ax.set_xlabel(r"$N$", fontsize="x-large")
    ax.set_ylabel(r"$P$", fontsize="x-large")

    plt.suptitle(r"$\sigma=0,D_\psi=0$",fontsize="xx-large")

    ax.set_ylim(0.35, 1)
    add_line(ax, 0, 0.935, 1, -0.0062, label=r"$N^{-0.0062}$", yl=0.85)

    # add_line(ax, 0, 1, 1, -1/16, label=r"$N^{-1/16}$", xl=0.65, yl=0.85)
    # add_line(ax, 0, 0.83, 1, -1/16)
    add_line(ax, 0, 0.552, 1, -1/19, label=r"$N^{-1/19}$", yl=0.4)
    # add_line(ax, 0, 0.845, 1, -1/19)
    # add_line(ax, 0, 1, 1, -1/2, label=r"$N^{-1/2}$", yl=0.45)

    plt.show()
    plt.close()


if __name__ == "__main__":
    rho0 = 1
    D_psi = 0.1  # temperature for spins
    sigma = 0.1
    L = 4096
    D_theta = 0.
    seed = 3000
    t_arr, phi_arr, theta_arr = read_op_series(L, rho0, D_psi, sigma, D_theta, seed=seed, ret_theta=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    ax1.plot(t_arr, phi_arr)
    ax2.plot(t_arr, theta_arr)

    # ax1.set_xscale("log")
    # ax1.set_yscale("log")
    # add_line(ax1, 0, 1, 1, -0.00625, label=r"$N^{-1/16}$", xl=0.65, yl=0.85)

    plt.show()
    plt.close()
   
    # varied_D_psi()
    # varied_sigma()