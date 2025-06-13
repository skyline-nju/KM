import numpy as np
import matplotlib.pyplot as plt
from add_line import add_line


def varied_L(sigma=0.1, D=0, mode="theta"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    if mode == "theta":
        folder = "/mnt/sda/active_KM/snap/corr_func/untangled"
    else:
        folder = "/mnt/sda/active_KM/snap/corr_func"
    if sigma == 0.1:
        # L_arr = [1024, 2048, 2880, 4096, 5760]
        if D == 0:
            L_arr = [1024, 2048, 2880, 4096, 5760]
        elif D == 0.1:
            L_arr = [2048, 2880]
    else:
        L_arr = [2048, 4096]
    for L in L_arr:
        with np.load(f"{folder}/L{L:d}_{L:d}_r1_v1_T0.1_s{sigma:g}_D{D:.4f}_h0.1_S3000.npz") as data:
            q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
        ax1.plot(q, Sq, "-o", label=r"$L=%d$" % L)
        ax2.plot(r, Cr/L, "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")
        ax3.plot(r, Cr, "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")

    ax1.set_xlabel(r"$k$", fontsize="x-large")
    ax2.set_xlabel(r"$r$", fontsize="x-large")
    ax1.set_ylabel(r"$\langle |\tilde{\mathbf{u}}(\mathbf{k})|^2\rangle$", fontsize="x-large")
    ax2.set_ylabel(r"$\langle \mathbf{u}(0)\cdot \mathbf{u}(\mathbf{r})\rangle$", fontsize="x-large")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    # ax1.set_xlim(0.001)
    ax1.legend(fontsize="x-large")
    fig.suptitle(r"$T=0.1, \sigma=%g, D_\psi=0$" % (sigma), fontsize="xx-large")
    # ax2.set_ylim(1e-3)
    # ax3.set_ylim(1e-3)
    # ax2.set_xlim(5)
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 0.5, 1, -1, label=r"$-1$")
    add_line(ax1, 0, 0.8, 1, -3, label=r"$-3$")
    add_line(ax2, 0, 1, 1, -0.25, label=r"$-1/4$", yl=0.6)
    # add_line(ax2, 0.5, 1, 1, -0.6, label=r"$-0.6$", yl=0.8, xl=0.55)
    add_line(ax2, 0.6, 1, 1, -0.93, label=r"$-0.93$", yl=0.8, xl=0.7)
    plt.show()
    plt.close()


def varied_sigma(L=2048):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    folder = "/mnt/sda/active_KM/snap"
    with np.load(f"{folder}/corr_func/L{L:d}_{L:d}_r1_v1_T0.1_s0.1_D0.0000_h0.1_S3000.npz") as data:
        q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
    ax1.plot(q, Sq, label=r"$\sigma=0.1$", c="tab:green")
    ax2.plot(r, Cr, "-o", label=r"$\sigma=0.1$", c="tab:green")
    ax3.plot(r, Cr, "-o", label=r"$\sigma=0.1$", c="tab:green")

    with np.load(f"{folder}/corr_func/L{L:d}_{L:d}_r1_v1_T0.1_s0.2_D0.0000_h0.1_S3000.npz") as data:
        q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
    ax1.plot(q, Sq, label=r"$\sigma=0.2$", c="tab:red")
    ax2.plot(r, Cr, "-o", label=r"$\sigma=0.2$", c="tab:red")
    ax3.plot(r, Cr, "-o", label=r"$\sigma=0.2$", c="tab:red")

    ax1.set_xlabel(r"$k$", fontsize="x-large")
    ax2.set_xlabel(r"$r$", fontsize="x-large")
    ax3.set_xlabel(r"$r$", fontsize="x-large")
    ax1.set_ylabel(r"$\langle |\tilde{\mathbf{u}}(\mathbf{k})|^2\rangle$", fontsize="x-large")
    ax2.set_ylabel(r"$\langle \mathbf{u}(0)\cdot \mathbf{u}(\mathbf{r})\rangle$", fontsize="x-large")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    # ax2.set_xscale("log")
    ax3.set_yscale("log")
    ax1.set_xlim(0.005)
    ax1.legend(fontsize="x-large")
    fig.suptitle(r"$T=0.1, D_\psi=0, L=%d$" % L, fontsize="xx-large")
    ax2.set_ylim(1e-3)
    ax3.set_ylim(1e-4)
    # ax2.set_xlim(5)
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 0.5, 1, -1, label=r"$-1$")
    add_line(ax2, 0, 1, 1, -0.25, label=r"$-1/4$", yl=0.93)
    add_line(ax2, 0.5, 1, 1, -0.6, label=r"$-0.6$", yl=0.8, xl=0.8)
    plt.show()
    plt.close()


def varied_D():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    folder = "/mnt/sda/active_KM/snap"

    L = 2048
    T = 0.1
    sigma = 0.1
    D_arr = [0, 0.001, 0.01]
    for D in D_arr:
        with np.load(f"{folder}/corr_func/L{L:d}_{L:d}_r1_v1_T{T:g}_s{sigma:g}_D{D:.4f}_h0.1_S3000.npz") as data:
            q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
        ax1.plot(q, Sq, label=r"$D=%g$" % D)
        ax2.plot(r, Cr, "-", label=r"$D=%g$" % D, ms=3, fillstyle="none")
        ax3.plot(r, Cr, "-", label=r"$D=%g$" % D, ms=3, fillstyle="none")

    ax1.set_xlabel(r"$k$", fontsize="x-large")
    ax2.set_xlabel(r"$r$", fontsize="x-large")
    ax1.set_ylabel(r"$\langle |\tilde{\mathbf{u}}(\mathbf{k})|^2\rangle$", fontsize="x-large")
    ax2.set_ylabel(r"$\langle \mathbf{u}(0)\cdot \mathbf{u}(\mathbf{r})\rangle$", fontsize="x-large")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax1.set_xlim(0.001)
    ax1.legend(fontsize="x-large")
    fig.suptitle(r"$L=%d, T=0.1, \sigma=%g$" % (L, sigma), fontsize="xx-large")
    ax2.set_ylim(3e-1, 1)
    ax3.set_ylim(1e-3, 1)
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 0.5, 1, -1, label=r"$-1$")
    add_line(ax2, 0, 1, 1, -0.25, label=r"$-1/4$", yl=0.6)
    add_line(ax2, 0.5, 1, 1, -0.6, label=r"$-0.6$", yl=0.8, xl=0.55)
    # add_line(ax2, 0.6, 1, 1, -0.93, label=r"$-0.93$", yl=0.8, xl=0.55)
    plt.show()
    plt.close()


def corr_u_vs_corr_theta(sigma=0.1):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    sigma = 0.1
    L = 2880 * 2
    D = 0
    # L = 4096
    seed = 3000
    folders = ["/mnt/sda/active_KM/snap/corr_func", "/mnt/sda/active_KM/snap/corr_func/untangled"]
    for folder in folders:
        with np.load(f"{folder}/L{L:d}_{L:d}_r1_v1_T0.1_s{sigma:g}_D{D:.4f}_h0.1_S{seed}.npz") as data:
            q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
        print(q)
        ax1.plot(q, Sq, "-o")
        ax2.plot(r, Cr/Cr[0], "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")
        ax3.plot(r, Cr/Cr[0], "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")

    ax1.set_xlabel(r"$k$", fontsize="x-large")
    ax2.set_xlabel(r"$r$", fontsize="x-large")
    ax1.set_ylabel(r"$\langle |\tilde{\mathbf{u}}(\mathbf{k})|^2\rangle$", fontsize="x-large")
    ax2.set_ylabel(r"$\langle \mathbf{u}(0)\cdot \mathbf{u}(\mathbf{r})\rangle$", fontsize="x-large")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    # ax1.set_xlim(0.001)
    # ax1.set_ylim(1)
    # ax1.legend(fontsize="x-large")
    fig.suptitle(r"$T=0.1, \sigma=%g, D_\psi=%g$" % (sigma, D), fontsize="xx-large")
    ax2.set_ylim(1e-3, 1)
    ax3.set_ylim(1e-3, 1)
    # ax2.set_xlim(5)
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 1, 1, -4, label=r"$-4$", yl=0.5)
    add_line(ax1, 0, 0.5, 1, -1, label=r"$-1$")
    add_line(ax2, 0, 1, 1, -0.25, label=r"$-1/4$", yl=0.6)
    # add_line(ax2, 0.5, 1, 1, -0.6, label=r"$-0.6$", yl=0.8, xl=0.55)
    add_line(ax2, 0.6, 1, 1, -0.93, label=r"$-0.93$", yl=0.8, xl=0.7)
    plt.show()
    plt.close()


def varied_D_two_corr():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharex=True, sharey=True)

    sigma = 0.1
    L = 2048
    seed = 3000
    D_arr = [0, 0.001, 0.01, 0.1, 1]
    for D in D_arr:
    # L = 4096
        folders = ["/mnt/sda/active_KM/snap/corr_func", "/mnt/sda/active_KM/snap/corr_func/untangled"]
        axes = [ax1, ax2]
        for i, folder in enumerate(folders):
            with np.load(f"{folder}/L{L:d}_{L:d}_r1_v1_T0.1_s{sigma:g}_D{D:.4f}_h0.1_S{seed}.npz") as data:
                q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
            axes[i].plot(q, Sq, label=r"$D_r=%g$" % D)

        ax1.set_xlabel(r"$k$", fontsize="x-large")
        ax2.set_xlabel(r"$k$", fontsize="x-large")
        ax1.set_ylabel(r"$\langle |\tilde{\mathbf{u}}(\mathbf{k})|^2\rangle$", fontsize="x-large")
        ax2.set_ylabel(r"$\langle |\tilde{\theta} (\mathbf{k})|^2\rangle$", fontsize="x-large")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        # ax1.set_xlim(0.001)
        # ax1.set_ylim(1)
        # ax1.legend(fontsize="x-large")
        fig.suptitle(r"$T=0.1, \sigma=%g, L=%g$" % (sigma, L), fontsize="xx-large")
        # ax2.set_xlim(5)
    ax1.legend(fontsize="large")
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 1, 1, -4, label=r"$-4$", yl=0.6)
    add_line(ax2, 0, 1, 1, -2, label=r"$-2$", yl=0.8)
    add_line(ax2, 0, 1, 1, -4, label=r"$-4$", yl=0.6)
    # add_line(ax2, 0, 0.8, 1, -3, label=r"$-3$", yl=0.3)
    plt.show()
    plt.close()


def varied_L_two_corr():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharex=True, sharey=True)

    sigma = 0.
    D = 0
    seed = 1000
    L_arr = [2048]
    for L in L_arr:
    # L = 4096
        folders = ["/mnt/sda/active_KM/snap/corr_func", "/mnt/sda/active_KM/snap/corr_func/untangled"]
        axes = [ax1, ax2]
        for i, folder in enumerate(folders):
            with np.load(f"{folder}/L{L:d}_{L:d}_r1_v1_T0.1_s{sigma:g}_D{D:.4f}_h0.1_S{seed}.npz") as data:
                q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
            axes[i].plot(q, Sq,"-o", label=r"$L=%g$" % L, fillstyle="none", ms=3)

        ax1.set_xlabel(r"$k$", fontsize="x-large")
        ax2.set_xlabel(r"$k$", fontsize="x-large")
        ax1.set_ylabel(r"$\langle |\tilde{\mathbf{u}}(\mathbf{k})|^2\rangle$", fontsize="x-large")
        ax2.set_ylabel(r"$\langle |\tilde{\theta} (\mathbf{k})|^2\rangle$", fontsize="x-large")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        # ax1.set_xlim(0.001)
        # ax1.set_ylim(1)
        # ax1.legend(fontsize="x-large")
        fig.suptitle(r"$T=0.1, \sigma=%g, D_r=%g$" % (sigma, D), fontsize="xx-large")
        # ax2.set_xlim(5)
    ax1.legend(fontsize="large")
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 1, 1, -4, label=r"$-4$", yl=0.6)
    add_line(ax2, 0, 1, 1, -2, label=r"$-2$", yl=0.8)
    add_line(ax2, 0, 1, 1, -4, label=r"$-4$", yl=0.4)
    add_line(ax2, 0, 0.94, 1, -3, label=r"$-3$", yl=0.15, xl=0.6)
    plt.show()
    plt.close()

if __name__ == "__main__":
    # varied_L(sigma=0.1, D=0.1, mode="theta")
    # varied_sigma(L=4096)

    # varied_D()
    # corr_u_vs_corr_theta()
    varied_L_two_corr()

