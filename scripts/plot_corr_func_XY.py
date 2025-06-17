import numpy as np
import matplotlib.pyplot as plt
from add_line import add_line


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), constrained_layout=True)

    folder = "/mnt/sda/active_KM/snap/corr_func"

    L = 2048
    T = 0.1
    D = 0
    sigma = 0
    seed = 1000

    fname = f"{folder}/L{L:d}_{L:d}_r1_v1_T{T:g}_s{sigma:g}_D{D:.4f}_h0.1_S{seed}.npz"

    with np.load(fname) as data:
        q, Sq, r, Cr = data["q"], data["Sq"], data["r"], data["Cr"]
        ax1.plot(q, Sq, "-o", label=r"$L=%d$" % L, ms=3, fillstyle="none")
        ax2.plot(r, Cr, "-o", label=r"$L=%d$" % L, ms=3, fillstyle="none")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax1.set_xlabel(r"$k$", fontsize="x-large")
    ax2.set_xlabel(r"$r$", fontsize="x-large")
    ax1.set_ylabel(r"$\langle|\tilde{\mathbf{u}}(\mathbf{k})|^2 \rangle$", fontsize="x-large")
    ax2.set_ylabel(r"$\langle \mathbf{u}(0)\cdot \mathbf{u}(\mathbf{r}) \rangle$", fontsize="x-large")

    plt.suptitle(r"$L=%d, T=%g, \sigma=%g, D_\psi=%g$" % (L, T, sigma, D), fontsize="x-large")

    add_line(ax1, 0, 1, 1, -2, label=r"$-2$")
    add_line(ax2, 0, 0.85, 1, -0.0248, label=r"$-0.0248$")
    plt.show()
    plt.close()