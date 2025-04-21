#include <iostream>
#include "domain2D.h"
#include "cellList2D.h"
#include "particle2D.h"
#include "force2D.h"
#include "integrate2D.h"
#include "exporter2D.h"
#include "comn.h"

int main(int argc, char* argv[]) {
  double Lx = 256;
  double Ly = 256;

  double rho0 = atof(argv[1]);
  double T = atof(argv[2]);
  double v0 = 1;
  double D_theta = atof(argv[3]);
  double sigma = atof(argv[4]);

  double h = atof(argv[5]);
  int n_step = atoi(argv[6]);
  //int snap_interval = int(round(200 / h * 0.1));
  double snap_log_sep = atof(argv[7]);

  int seed = atoi(argv[8]);
  std::string ini_mode = argv[9];  // should be "bimodal" or "resume"

  int snap_interval = 1;
  if (snap_log_sep > 1) {
    snap_interval = int(snap_log_sep);
    snap_log_sep = -1;
  }
  int n_par = int(rho0 * Lx * Ly);

  typedef BiNode<ActiveOscillator_2> node_t;
  Ranq2 myran(seed);
  Vec_2<double> gl_l(Lx, Ly);
  double r_cut = 1;
  Grid_2 grid(gl_l, r_cut);
  PeriodicDomain_2 pdm(gl_l);
  CellListNode_2<node_t> cl(pdm, grid);
  std::vector<node_t> p_arr;

  // ini integrator
  MotileOscillatorEM integrator(h, D_theta, T, v0);

  // cal force
  AlignKernal kernal(r_cut);
  auto f1 = [&kernal](node_t* p1, node_t* p2) {
    kernal(*p1, *p2);
    };
  auto f2 = [&kernal, &pdm](node_t* p1, node_t* p2) {
    kernal(*p1, *p2, pdm);
    };

  // set output
  char basename[255];
  char log_file[255];
  char op_file[255];
  char gsd_file[255];
#ifdef _MSC_VER
  char folder[] = "data\\";
#else
  char folder[] = "/mnt/d/code/KM/data/";
#endif

  char log_folder[255];
  char op_folder[255];
  snprintf(log_folder, 255, "%slog%s", folder, delimiter.c_str());
  snprintf(op_folder, 255, "%sop%s", folder, delimiter.c_str());
  mkdir(log_folder);
  mkdir(op_folder);

  snprintf(basename, 255, "L%g_%g_r%g_v%g_T%g_s%g_D%.4f_h%g_S%d",
    Lx, Ly, rho0, v0, T, sigma, D_theta, h, seed);
  snprintf(gsd_file, 255, "%s%s.gsd", folder, basename);

  int start = 0;

  exporter::Snap_GSD_2 gsd(gsd_file, n_step, snap_interval, start, h, snap_log_sep, gl_l, ini_mode);

  int log_interval = 10000;
  int op_interval = 100;
  snprintf(log_file, 255, "%s%s_t%d.dat", log_folder, basename, start);
  exporter::LogExporter log(log_file, start, n_step, log_interval, n_par);

  snprintf(op_file, 255, "%s%s_t%d.dat", op_folder, basename, start);
  exporter::OrderParaExporter op(op_file, start, n_step, op_interval);

  // ini particles
  ini_particles(p_arr, myran, ini_mode, n_par, gl_l, sigma, gsd);
  cl.create(p_arr);

  for (int t = 1; t <= n_step; t++) {
    cl.for_each_pair(f1, f2);
    for (int i = 0; i < n_par; i++) {
      integrator.update(p_arr[i], pdm, myran);
      //integrator.update_par_cellList(p_arr[i], pdm, myran, cl);
    }
    cl.recreate(p_arr);
    gsd.dump(t, p_arr);
    op.dump(t, p_arr);
    log.record(t);
  }
}
