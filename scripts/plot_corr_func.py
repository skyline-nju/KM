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
            L_arr = [2048, 4096]
    elif sigma == 0.025:
        if D == 0.1:
            L_arr = [2048, 4096]
    else:
        L_arr = [2048, 4096]
    for L in L_arr:
        with np.load(f"{folder}/L{L:d}_{L:d}_r1_v1_T0.1_s{sigma:g}_D{D:.4f}_h0.1_S3000.npz") as data:
            q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
            if "Gr" in data:
                Gr = data["Gr"]

        ax1.plot(q, Sq, "-o", label=r"$L=%d$" % L)
        ax2.plot(r, Cr/L, "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")
        ax3.plot(r, Cr/L, "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")

    ax1.set_xlabel(r"$k$", fontsize="x-large")
    ax2.set_xlabel(r"$r$", fontsize="x-large")
    if mode == "u":
        ax1.set_ylabel(r"$\langle |\tilde{\mathbf{u}}(\mathbf{k})|^2\rangle$", fontsize="x-large")
        ax2.set_ylabel(r"$\langle \mathbf{u}(0)\cdot \mathbf{u}(\mathbf{r})\rangle$", fontsize="x-large")
    else:
        ax1.set_ylabel(r"$\langle|\tilde{\theta}(\mathbf{k})|^2 \rangle $", fontsize="x-large")
        ax2.set_ylabel(r"$\langle \theta(0)\theta(\mathbf{r})\rangle$", fontsize="x-large")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    # ax1.set_xlim(0.001)
    ax1.legend(fontsize="x-large")
    fig.suptitle(r"$T=0.1, \sigma=%g, D_\psi=%g$" % (sigma, D), fontsize="xx-large")
    # ax2.set_ylim(1e-3)
    # ax3.set_ylim(1e-3)
    # ax2.set_xlim(5)
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    # add_line(ax1, 0, 0.5, 1, -1, label=r"$-1$")
    add_line(ax1, 0, 0.85, 1, -4, label=r"$-4$")
    add_line(ax2, 0, 1, 1, -0.25, label=r"$-1/4$", yl=0.6)
    add_line(ax2, 0.6, 1, 1, -0.93, label=r"$-0.93$", yl=0.8, xl=0.7)
    add_line(ax2, 0., 0, 1, 1.5, label=r"$3$", yl=0.8, xl=0.55)
    plt.show()
    plt.close()


def varied_L_varied_corr(sigma=0.1, D=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)

    folder = "/mnt/sda/active_KM/snap/corr_func/untangled"
    if sigma == 0.1:
        # L_arr = [1024, 2048, 2880, 4096, 5760]
        if D == 0:
            L_arr = [1024, 2048, 2880, 4096, 5760]
        elif D == 0.1:
            L_arr = [2048, 2880, 4096]
    elif sigma == 0.025:
        if D == 0.1:
            L_arr = [2048, 4096]
    else:
        L_arr = [2048, 4096]
    for L in L_arr:
        with np.load(f"{folder}/L{L:d}_{L:d}_r1_v1_T0.1_s{sigma:g}_D{D:.4f}_h0.1_S3000.npz") as data:
            q, Sq, r, Cr, Gr, Cr_raw = data["q"], data["Sq"], data["r"], data["Cr"], data["Gr"], data["Cr_raw"]

        axes[0, 0].plot(q, Sq, "-", label=r"$L=%d$" % L)
        axes[0, 1].plot(r, Cr_raw/L, "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")
        axes[1, 0].plot(r, Cr/L, "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")
        axes[1, 1].plot(r[1:], Gr[1:]/L, "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")

    axes[0, 0].set_xlabel(r"$k$", fontsize="x-large")
    axes[0, 1].set_xlabel(r"$r$", fontsize="x-large")
    axes[1, 0].set_xlabel(r"$r$", fontsize="x-large")
    axes[1, 1].set_xlabel(r"$r$", fontsize="x-large")
    axes[0, 0].set_ylabel(r"$\langle|\tilde{\theta}(\mathbf{k})|^2 \rangle $", fontsize="x-large")
    axes[0, 1].set_ylabel(r"$\langle \theta(0)\theta(\mathbf{r})\rangle$", fontsize="x-large")
    axes[1, 0].set_ylabel(r"$\langle (\theta(0)-\bar{\theta}) (\theta(\mathbf{r})-\bar{\theta})\rangle$", fontsize="x-large")
    axes[1, 1].set_ylabel(r"$\langle \left[\theta(0)-\theta(\mathbf{r})\right ]^2 \rangle$", fontsize="x-large")

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0, 0].legend(fontsize="x-large")
    fig.suptitle(r"$T=0.1, \sigma=%g, D_\psi=%g$" % (sigma, D), fontsize="xx-large")
    add_line(axes[0, 0], 0, 0.85, 1, -4, label=r"$-4$")
    add_line(axes[1, 1], 0.1, 0.05, 1, 7/4, label=r"$\frac{7}{4}$", yl=0.55)
    add_line(axes[1, 1], 0.1, 0.05, 1, 3/2, label=r"$\frac{3}{2}$", yl=0.3, xl=0.4)
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


def corr_u_vs_corr_theta():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    T = 0.1
    sigma = 0.
    L = 2048
    D = 0.
    # L = 4096
    seed = 1000
    folders = ["/mnt/sda/active_KM/snap/corr_func", "/mnt/sda/active_KM/snap/corr_func/untangled"]
    for folder in folders:
        with np.load(f"{folder}/L{L:d}_{L:d}_r1_v1_T{T:g}_s{sigma:g}_D{D:.4f}_h0.1_S{seed}.npz") as data:
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
            axes[i].plot(q, Sq, label=r"$D_\psi=%g$" % D)

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5), constrained_layout=True, sharex=True, sharey=True)

    sigma = 0.025
    D = 0.1
    seed = 3000
    if sigma == 0.1:
        L_arr = [2048, 2880, 4096]
    elif sigma == 0.05:
        L_arr = [2048, 4096]
    elif sigma == 0.025:
        L_arr = [2048, 4096]
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
        fig.suptitle(r"$T=0.1, \sigma=%g, D_\psi=%g$" % (sigma, D), fontsize="xx-large")
        # ax2.set_xlim(5)
    ax1.legend(fontsize="large")
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 1, 1, -4, label=r"$-4$", yl=0.6)
    add_line(ax2, 0, 1, 1, -2, label=r"$-2$", yl=0.8)
    add_line(ax2, 0, 1, 1, -4, label=r"$-4$", yl=0.4)
    # add_line(ax2, 0, 0.96, 1, -3, label=r"$-3$", yl=0.15, xl=0.6)
    plt.show()
    plt.close()


def varied_sigma_two_corr():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.5), constrained_layout=True, sharex=True, sharey=True)

    L = 2048
    D = 0.1
    if D == 0.:
        seed_arr = [1000, 3000, 3000]
        sigma_arr = [0, 0.05, 0.1]
    elif D == 0.1:
        seed_arr = [3000, 3000, 3000]
        sigma_arr = [0.025, 0.05, 0.1]
    for j, sigma in enumerate(sigma_arr):
        folders = ["/mnt/sda/active_KM/snap/corr_func", "/mnt/sda/active_KM/snap/corr_func/untangled"]
        axes = [ax1, ax2]
        for i, folder in enumerate(folders):
            with np.load(f"{folder}/L{L:d}_{L:d}_r1_v1_T0.1_s{sigma:g}_D{D:.4f}_h0.1_S{seed_arr[j]}.npz") as data:
                q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
            axes[i].plot(q, Sq,"-o", label=r"$\sigma=%g$" % sigma, fillstyle="none", ms=3)

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
        fig.suptitle(r"$T=0.1, L=%g, D_\psi=%g$" % (L, D), fontsize="xx-large")
        # ax2.set_xlim(5)
    ax1.legend(fontsize="large")
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 1, 1, -4, label=r"$-4$", yl=0.6)
    add_line(ax2, 0, 1, 1, -2, label=r"$-2$", yl=0.8)
    add_line(ax2, 0, 1, 1, -4, label=r"$-4$", yl=0.6)
    add_line(ax2, 0, 0.6, 1, -2, label=r"$-2$", yl=0.2, xl=0.5)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # varied_L(sigma=0.1, D=0.1, mode="theta")
    varied_L_varied_corr(sigma=0.1, D=0.1)
    # varied_sigma(L=4096)

    # varied_D()
    # corr_u_vs_corr_theta()
    # varied_L_two_corr()
    # varied_D_two_corr()
    # varied_sigma_two_corr()
