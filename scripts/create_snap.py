import sys
import numpy as np
from gsd import hoomd
import os


def duplicate(s, nx: int, ny: int):
    N = s.particles.N * nx * ny
    lx = s.configuration.box[0]
    ly = s.configuration.box[1]
    Lx, Ly = lx * nx, ly * ny
    pos = np.zeros((N, 3), dtype=np.float32)
    charge = np.zeros(N, dtype=np.float32)
    mass = np.zeros(N, dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            beg = (j * nx + i) * s.particles.N
            end = beg + s.particles.N
            pos[beg:end, 0] = s.particles.position[:, 0] + lx / 2 + i * lx
            pos[beg:end, 1] = s.particles.position[:, 1] + ly / 2 + j * ly
            pos[beg:end, 2] = s.particles.position[:, 2]
            charge[beg:end] = s.particles.charge
            mass[beg:end] = s.particles.mass
    pos[:, 0] -= Lx / 2
    pos[:, 1] -= Ly / 2
    s2 = hoomd.Frame()
    s2.configuration.box = [Lx, Ly, 1, 0, 0, 0]
    s2.particles.N = N
    s2.particles.position = pos
    s2.particles.charge = charge
    s2.particles.mass = mass
    s2.configuration.step = 0
    return s2


def create_spin_waves(angle, L, rho0=1, n_period=1):
    # angle <= pi/4
    s = hoomd.Frame()
    s.configuration.box = [L, L, 1, 0, 0, 0]
    N = int(L ** 2 * rho0)
    s.particles.N = N
    s.configuration.step = 0

    pos = np.zeros((N, 3), dtype=np.float32)
    charge = np.zeros(N, dtype=np.float32)
    mass = np.zeros(N, dtype=np.float32)

    pos[:, 0] = np.random.rand(N) * L
    pos[:, 1] = np.random.rand(N) * L
    pos[:, 2] = (np.random.rand(N) - 0.5)  * np.pi * 2

    z = pos[:, 0] + pos[:, 1] * np.tan(angle)
    z[z>= L] -= L
    charge = 2 * np.pi * z / L
    charge[charge >= np.pi] -= 2 * np.pi

    pos[:, 0] -= L / 2
    pos[:, 1] -= L / 2
    s.particles.position = pos
    s.particles.charge = charge
    s.particles.mass = mass
    return s


def create_spin_waves_along_y(Lx, Ly, rho0=1, n_period=1):
    s = hoomd.Frame()
    s.configuration.box = [Lx, Ly, 1, 0, 0, 0]
    N = int(Lx * Ly * rho0)
    s.particles.N = N
    s.configuration.step = 0

    pos = np.zeros((N, 3), dtype=np.float32)
    charge = np.zeros(N, dtype=np.float32)
    mass = np.zeros(N, dtype=np.float32)

    pos[:, 0] = np.random.rand(N) * Lx
    pos[:, 1] = np.random.rand(N) * Ly
    pos[:, 2] = (np.random.rand(N) - 0.5)  * np.pi * 2

    charge = 2 * np.pi * pos[:, 1] / Ly
    charge[charge >= np.pi] -= 2 * np.pi

    pos[:, 0] -= Lx / 2
    pos[:, 1] -= Ly / 2
    s.particles.position = pos
    s.particles.charge = charge
    s.particles.mass = mass
    return s


def scale(s, nx: int, ny: int, eps=0):
    lx = s.configuration.box[0]
    ly = s.configuration.box[1]
    Lx, Ly = lx * nx, ly * ny
    if isinstance(nx, int) and isinstance(ny, int):
        N = s.particles.N * nx * ny
        pos = np.zeros((N, 3), dtype=np.float32)
        type_id = np.zeros(N, dtype=np.uint32)
        for i in range(nx * ny):
            beg = i * s.particles.N
            end = beg + s.particles.N
            pos[beg:end, 0] = s.particles.position[:, 0] * nx
            pos[beg:end, 1] = s.particles.position[:, 1] * ny
            pos[beg:end, 2] = s.particles.position[:, 2]
            type_id[beg:end] = s.particles.typeid
        if nx > 1:
            pos[:, 0] += (np.random.rand(N) - 0.5) * eps * nx
            mask = pos[:, 0] < Lx/2
            pos[:, 0][mask] += Lx
            mask = pos[:, 0] >= Lx/2
            pos[:, 0][mask] -= Lx
        if ny > 1:
            pos[:, 1] += (np.random.rand(N) - 0.5) * eps * ny
            mask = pos[:, 1] < Ly/2
            pos[:, 1][mask] += Ly
            mask = pos[:, 1] >= Ly/2
            pos[:, 1][mask] -= Ly
        s2 = hoomd.Frame()
        s2.configuration.box = [Lx, Ly, 1, 0, 0, 0]
        s2.particles.N = N
        s2.particles.position = pos
        s2.particles.typeid = type_id
        s2.particles.types = s.particles.types
        s2.configuration.step = 0
    else:
        s.particles.position[:, 0] *= nx
        s.particles.position[:, 1] *= ny
        s.configuration.box = [Lx, Ly, 1, 0, 0, 0]

        mask_A = s.particles.typeid == 0
        mask_B = s.particles.typeid == 1
        rho_A = np.sum(mask_A) / (lx * ly)
        rho_B = np.sum(mask_B) / (lx * ly)
        print("rho_A=", rho_A, "rho_B=", rho_B)
        s2 = adjust_density(s, rho_A, rho_B, mode="copy")
    return s2


if __name__ == "__main__":
    folder = r"/mnt/sda/active_KM/finite_PD"
    # folder = "build/data"
    basename = "L1024_128_r2.5_v0_T0.1_J2.5_s0_D0.0000_h0.1_S2000.gsd"

    fname_in = f"{folder}/{basename}"
    # with hoomd.open(name=fname_in, mode='r') as fin:
    #     print(len(fin))
    #     snap = fin[-1]

    fname_out = fname_in
    with hoomd.open(name=fname_out, mode='w') as fout:
        # snap_new = duplicate(snap, 2, 2)
        snap_new = create_spin_waves_along_y(1024, 128, 2.5)
        fout.append(snap_new)

