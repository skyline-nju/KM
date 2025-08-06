import numpy as np
import os
import sys
import glob
from gsd import fl
import matplotlib.pyplot as plt


def coarse_grain(xs, ys, angles, Lx, Ly, dx=4):
    nx = Lx // dx
    ny = Ly // dx

    xs += Lx / 2
    ys += Ly / 2

    xs[xs < 0] += Lx
    xs[xs >= Lx] -= Lx
    ys[ys < 0] += Ly
    ys[ys >= Ly] -= Ly

    xs_new = xs / dx
    ys_new = ys / dx

    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    ux = np.zeros((ny, nx), np.float32)
    uy = np.zeros((ny, nx), np.float32)
    num = np.zeros((ny, nx), np.int32)

    for i in range(xs.size):
        xi = int(xs_new[i])
        yi = int(ys_new[i])
        ux[yi, xi] += cos_angles[i]
        uy[yi, xi] += sin_angles[i]
        num[yi, xi] += 1
    return ux, uy, num


'''
show the x-component of global polarity
'''
def plot_polarity_x():
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fname = r"D:\tmp\L128_128_r2.5_v1_T0.1_J1_s0_D0.0000_h0.1_S2001.gsd"
    with fl.open(name=fname, mode="r") as fin:
        nframes = fin.nframes
        box = fin.read_chunk(frame=0, name="configuration/box")
        Lx, Ly = int(box[0]), int(box[1])

        px = np.zeros(nframes)
        for i_frame in range(nframes):
            pos = fin.read_chunk(frame=i_frame, name="particles/position")
            theta = pos[:, 2]
            px[i_frame] = np.mean(np.cos(theta))
        
    
    t = np.arange(nframes) * 1000
    ax.plot(t, px, label=r"$D_\psi=0$")

    fname = r"D:\tmp\L128_128_r2.5_v1_T0.1_J1_s0_D0.1000_h0.1_S2001.gsd"
    with fl.open(name=fname, mode="r") as fin:
        nframes = fin.nframes
        box = fin.read_chunk(frame=0, name="configuration/box")
        Lx, Ly = int(box[0]), int(box[1])

        px = np.zeros(nframes)
        for i_frame in range(nframes):
            pos = fin.read_chunk(frame=i_frame, name="particles/position")
            theta = pos[:, 2]
            px[i_frame] = np.mean(np.cos(theta))
        
    
    t = np.arange(nframes) * 1000
    ax.plot(t, px, label=r"$D_\psi=0.1$")

    ax.set_xlabel(r"$t$", fontsize="xx-large")
    ax.set_ylabel(r"$\langle \cos(\psi_i^t)\rangle_i $", fontsize="xx-large")
    ax.set_xlim(0, 6e5)
    ax.legend(fontsize="xx-large")
    plt.show()
    plt.close()


def get_para(fname, suffix=".gsd"):
    basename = os.path.basename(fname)
    s = basename.rstrip(suffix).split("_")
    para = {}
    if len(s) == 9:
        para["Lx"] = int(s[0].lstrip("L"))
        para["Ly"] = int(s[1])
        para["rho0"] = float(s[2].lstrip("r"))
        para["v0"] = float(s[3].lstrip("v"))
        para["T"] = float(s[4].lstrip("T"))
        para["sigma"] = float(s[5].lstrip("s"))
        para["D_theta"] = float(s[6].lstrip("D"))
        para["h"] = float(s[7].lstrip("h"))
        para["seed"] = int(s[8].lstrip("S"))
    else:
        print("length of para string=", len(s))
        sys.exit(1)
    return para


def get_coarse_grained_snaps(fname_in=None, para=None, dx=4, folder="/mnt/sda/active_KM/snap"):
    if fname_in is not None:
        para = get_para(fname_in)
    else:
        fname_in = f"{folder}/L%d_%d_r%g_v%g_T%g_s%g_D%.4f_h%g_S%d.gsd" % (
            para["Lx"], para["Ly"], para["rho0"], para["v0"], para["T"], para["sigma"], para["D_theta"],
            para["h"], para["seed"]
        )
    fname_out = f"{folder}/cg_dx{dx:d}/{os.path.basename(fname_in).replace(".gsd", ".npz")}"

    if os.path.exists(fname_out):
        with np.load(fname_out, "r") as data:
            ux0, uy0, num0, t0 = data["ux"], data["uy"], data["num"], data["t"]
    else:
        ux0, uy0, num0, t0 = None, None, None, None
    with fl.open(name=fname_in, mode="r") as fin:
        nframes = fin.nframes

        nrows, ncols = para["Ly"] // dx, para["Lx"] // dx
        ux = np.zeros((nframes, nrows, ncols), np.float32)
        uy = np.zeros((nframes, nrows, ncols), np.float32)
        num = np.zeros((nframes, nrows, ncols), np.int32)
        t = np.zeros(nframes)
        
        if ux0 is not None:
            n_existed = ux0.shape[0]
            ux[:n_existed] = ux0
            uy[:n_existed] = uy0
            num[:n_existed] = num0
            t[:n_existed] = t0
        else:
            n_existed = 0
        if n_existed < nframes:
            for i_frame in range(n_existed, nframes):
                print("frame, %d/%d" % (i_frame, nframes))
                pos = fin.read_chunk(frame=i_frame, name="particles/position")
                t[i_frame] = fin.read_chunk(frame=i_frame, name="configuration/step")[0] * para["h"]
                xs, ys = pos[:, 0], pos[0:, 1]
                angles = fin.read_chunk(frame=i_frame, name="particles/charge")
                ux[i_frame], uy[i_frame], num[i_frame] = coarse_grain(xs, ys, angles, para["Lx"], para["Ly"], dx)
            np.savez_compressed(fname_out, t=t, ux=ux, uy=uy, num=num)
        else:
            print(fname_out, "is up to date")


def update_coarse_grained_fields(Lx, Ly=None, dx=4):
    if Ly is None:
        Ly = Lx
    folder = "/mnt/sda/active_KM/snap"
    pat = f"{folder}/L{Lx:g}_{Ly:g}_*.gsd"
    files = glob.glob(pat)
    for file in files:
        get_coarse_grained_snaps(file, dx=dx, folder=folder)


if __name__ == "__main__":
    # folder = "/mnt/sda/active_KM/snap"
    # dx = 4

    # fname = f"{folder}/L2880_2880_r1_v1_T0.1_s0.1_D0.0000_h0.1_S3000.gsd"
    # get_coarse_grained_snaps(fname, dx=dx, folder=folder)
    update_coarse_grained_fields(Lx=4096)
