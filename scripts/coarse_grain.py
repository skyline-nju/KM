import numpy as np
import os
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


if __name__ == "__main__":
    folder = "/mnt/sda/active_KM/snap"
    L = 4096
    sigma = 0.1
    fname_in = f"{folder}/L{L:d}_{L:d}_r1_v1_T0.1_s{sigma:g}_D0.0000_h0.1_S3000.gsd"
    dx = 4

    with fl.open(name=fname_in, mode="r") as fin:
        nframes = fin.nframes
        box = fin.read_chunk(frame=0, name="configuration/box")
        Lx, Ly = int(box[0]), int(box[1])

        nrows, ncols = Ly // dx, Lx // dx
        ux = np.zeros((nframes, nrows, ncols), np.float32)
        uy = np.zeros((nframes, nrows, ncols), np.float32)
        num = np.zeros((nframes, nrows, ncols), np.int32)
        
        for i_frame in range(nframes):
            print("frame, %d/%d" % (i_frame, nframes))
            pos = fin.read_chunk(frame=i_frame, name="particles/position")
            xs, ys = pos[:, 0], pos[0:, 1]
            angles = fin.read_chunk(frame=i_frame, name="particles/charge")
            ux[i_frame], uy[i_frame], num[i_frame] = coarse_grain(xs, ys, angles, Lx, Ly, dx)

    fname_out = f"{folder}/cg_dx{dx:d}/{os.path.basename(fname_in).replace(".gsd", ".npz")}"
    np.savez_compressed(fname_out, ux=ux, uy=uy, num=num)