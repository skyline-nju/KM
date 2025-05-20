import numpy as np
import matplotlib.pyplot as plt
from add_line import add_line


def varied_L():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), constrained_layout=True)

    folder = "/mnt/sda/active_KM/snap"
    for L in [1024, 2048, 4096]:
        with np.load(f"{folder}/corr_func/L{L:d}_{L:d}_r1_v1_T0.1_s0.1_D0.0000_h0.1_S3000.npz") as data:
            q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
        ax1.plot(q, Sq, label=r"$L=%d$" % L)
        ax2.plot(r, Cr, "-", label=r"$L=%d$" % L, ms=3, fillstyle="none")

    ax1.set_xlabel(r"$k$", fontsize="x-large")
    ax2.set_xlabel(r"$r$", fontsize="x-large")
    ax1.set_ylabel(r"$\langle |\tilde{\mathbf{u}}(\mathbf{k})|^2\rangle$", fontsize="x-large")
    ax2.set_ylabel(r"$\langle \mathbf{u}(0)\cdot \mathbf{u}(\mathbf{r})\rangle$", fontsize="x-large")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax1.set_xlim(0.001)
    ax1.legend(fontsize="x-large")
    fig.suptitle(r"$T=0.1, \sigma=0.1, D_\psi=0$", fontsize="xx-large")
    # ax2.set_ylim(ymax=0.88)
    # ax2.set_xlim(5)
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 0.5, 1, -1, label=r"$-1$")
    add_line(ax2, 0, 1, 1, -0.25, label=r"$-1/4$", yl=0.6)
    # add_line(ax2, 0.5, 1, 1, -0.6, label=r"$-0.6$", yl=0.8, xl=0.55)
    add_line(ax2, 0.6, 1, 1, -0.93, label=r"$-0.93$", yl=0.8, xl=0.55)
    plt.show()
    plt.close()


def varied_sigma():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True)

    folder = "/mnt/sda/active_KM/snap"
    with np.load(f"{folder}/corr_func/L2048_2048_r1_v1_T0.1_s0.1_D0.0000_h0.1_S3000.npz") as data:
        q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
    ax1.plot(q, Sq, label=r"$\sigma=0.1$", c="tab:green")
    ax2.plot(r, Cr, "-o", label=r"$\sigma=0.1$", c="tab:green")
    ax3.plot(r, Cr, "-o", label=r"$\sigma=0.1$", c="tab:green")

    with np.load(f"{folder}/corr_func/L2048_2048_r1_v1_T0.1_s0.2_D0.0000_h0.1_S3000.npz") as data:
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
    fig.suptitle(r"$T=0.1, D_\psi=0, L=2048$", fontsize="xx-large")
    ax2.set_ylim(1e-3)
    ax3.set_ylim(1e-4)
    # ax2.set_xlim(5)
    add_line(ax1, 0, 1, 1, -2, label=r"$-2$", yl=0.7)
    add_line(ax1, 0, 0.5, 1, -1, label=r"$-1$")
    add_line(ax2, 0, 1, 1, -0.25, label=r"$-1/4$", yl=0.93)
    add_line(ax2, 0.5, 1, 1, -0.6, label=r"$-0.6$", yl=0.8, xl=0.8)
    plt.show()
    plt.close()


if __name__ == "__main__":
    varied_L()
    # varied_sigma()