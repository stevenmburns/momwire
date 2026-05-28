#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex.h>
#include "math.h"

#include <iostream>
#include <tuple>
#include <vector>

namespace py = pybind11;

// Batched cross-segment Gauss-Legendre quadrature in 3D.
//
// For each k in k_array, and each (i, j) segment pair, compute:
//   J_pq[k, i, j] = sum_{q, r} w_i_q * u_i_q^p * G(k, R_qr) * w_j_r * u_j_r^q
// where
//   R_qr = sqrt(|pos_i(t_q) - pos_j(t_r)|^2 + a_squared)
//   G(k, R) = exp(-j*k*R) / (4*pi*R)
//   pos_i(t) = (1-t) * seg_l_i + t * seg_r_i  (similarly for j)
//   w_i_q = len_i * gl_w_q;   u_i_q = len_i * gl_t_q
//
// a_squared = a^2 for V's off-edge case (corner regularization), 0 for the
// Yagi cross-wire case (no shared point so the kernel is already non-singular).
//
// Parallelism: each (i, j) pair is independent — OpenMP over the (i, j) grid.
// For each pair we precompute R[q, r] once, then loop k over all wavenumbers,
// reusing R from the per-thread cache.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
seg_seg_quad_batch_3d(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto ka  = k_array.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_k = ka.shape(0);
    size_t n_qp = glt.shape(0);

    py::array_t<std::complex<double>> J00({n_k, N_i, N_j});
    py::array_t<std::complex<double>> J10({n_k, N_i, N_j});
    py::array_t<std::complex<double>> J01({n_k, N_i, N_j});
    py::array_t<std::complex<double>> J11({n_k, N_i, N_j});
    auto j00 = J00.mutable_unchecked<3>();
    auto j10 = J10.mutable_unchecked<3>();
    auto j01 = J01.mutable_unchecked<3>();
    auto j11 = J11.mutable_unchecked<3>();

    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Per-segment quadrature-point positions and lengths -- k-independent so
    // compute once outside the parallel region.
    std::vector<double> pos_i(N_i * n_qp * 3);
    std::vector<double> pos_j(N_j * n_qp * 3);
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            pos_i[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
            pos_i[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
            pos_i[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
        }
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t r = 0; r < n_qp; r++) {
            double t = glt(r);
            pos_j[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
            pos_j[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
            pos_j[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            // Per-(i, j) R table — n_qp^2 doubles, lives on thread stack.
            double R[64];  // n_qp up to ~8 in practice (n_qp^2 = 64 max)
            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];
            for (size_t q = 0; q < n_qp; q++) {
                double pix = pi[q*3 + 0];
                double piy = pi[q*3 + 1];
                double piz = pi[q*3 + 2];
                for (size_t r = 0; r < n_qp; r++) {
                    double dx = pix - pj[r*3 + 0];
                    double dy = piy - pj[r*3 + 1];
                    double dz = piz - pj[r*3 + 2];
                    R[q*n_qp + r] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
            }

            double Li = len_i[i];
            double Lj = len_j[j];

            for (size_t kk = 0; kk < n_k; kk++) {
                double k = ka(kk);
                std::complex<double> s00(0, 0), s10(0, 0), s01(0, 0), s11(0, 0);
                for (size_t q = 0; q < n_qp; q++) {
                    double wi = glw(q) * Li;
                    double ui = glt(q) * Li;
                    for (size_t r = 0; r < n_qp; r++) {
                        double wj = glw(r) * Lj;
                        double uj = glt(r) * Lj;
                        double Rv = R[q*n_qp + r];
                        // G = exp(-j*k*R) / (4*pi*R)
                        double phase = -k * Rv;
                        std::complex<double> G(std::cos(phase), std::sin(phase));
                        G *= inv_4pi / Rv;
                        std::complex<double> wG = (wi * wj) * G;
                        s00 += wG;
                        s10 += ui * wG;
                        s01 += uj * wG;
                        s11 += (ui * uj) * wG;
                    }
                }
                j00(kk, i, j) = s00;
                j10(kk, i, j) = s10;
                j01(kk, i, j) = s01;
                j11(kk, i, j) = s11;
            }
        }
    }

    return std::make_tuple(J00, J10, J01, J11);
}


// Batched same-wire Gauss-Legendre quadrature on the *regularized* kernel
//   G_reg(R, k) = (exp(-j k R) - 1) / (4 pi R),    R = sqrt(diff^2 + a^2)
// for all segment pairs (i, j) on a shared 1D arc-length line.
//
// Inputs:
//   seg_endpoints : (N+1,) arc-length / position of segment boundaries
//   a             : wire radius (corner regularization)
//   k_array       : (n_k,) wavenumbers
//   gl_t, gl_w    : Gauss-Legendre nodes/weights mapped to [0, 1]
//                   (i.e. t = (xi + 1)/2 and w = w_legendre / 2)
//
// For each pair (i, j) we precompute the per-pair R table once and reuse it
// across the k axis. OpenMP parallelizes over (i, j).
//
// Output:
//   (J00, J10, J01, J11), each (n_k, N, N) complex.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
seg_seg_reg_quad_batch_1d(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_endpoints,
    double a,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    auto se  = seg_endpoints.unchecked<1>();
    auto ka  = k_array.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    if (se.shape(0) < 2) {
        throw std::runtime_error("seg_endpoints must have at least 2 entries");
    }

    size_t N    = se.shape(0) - 1;
    size_t n_k  = ka.shape(0);
    size_t n_qp = glt.shape(0);
    if (n_qp * n_qp > 64) {
        throw std::runtime_error("n_qp too large (n_qp^2 must be <= 64)");
    }

    const double a_sq    = a * a;
    const double inv_4pi = 1.0 / (4.0 * M_PI);

    py::array_t<std::complex<double>> J00({n_k, N, N});
    py::array_t<std::complex<double>> J10({n_k, N, N});
    py::array_t<std::complex<double>> J01({n_k, N, N});
    py::array_t<std::complex<double>> J11({n_k, N, N});
    auto j00 = J00.mutable_unchecked<3>();
    auto j10 = J10.mutable_unchecked<3>();
    auto j01 = J01.mutable_unchecked<3>();
    auto j11 = J11.mutable_unchecked<3>();

    // Per-segment quadrature positions and lengths.
    std::vector<double> pos(N * n_qp);
    std::vector<double> Lvec(N);
    for (size_t i = 0; i < N; i++) {
        double sl = se(i);
        double sr = se(i + 1);
        double Li = sr - sl;
        Lvec[i] = Li;
        for (size_t q = 0; q < n_qp; q++) {
            pos[i * n_qp + q] = sl + Li * glt(q);
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            double R[64];
            double Li = Lvec[i];
            double Lj = Lvec[j];
            const double *pi_q = &pos[i * n_qp];
            const double *pj_r = &pos[j * n_qp];
            for (size_t q = 0; q < n_qp; q++) {
                double piq = pi_q[q];
                for (size_t r = 0; r < n_qp; r++) {
                    double d = piq - pj_r[r];
                    R[q * n_qp + r] = std::sqrt(d * d + a_sq);
                }
            }

            for (size_t kk = 0; kk < n_k; kk++) {
                double k = ka(kk);
                std::complex<double> s00(0, 0), s10(0, 0), s01(0, 0), s11(0, 0);
                for (size_t q = 0; q < n_qp; q++) {
                    double wi = glw(q) * Li;
                    double ui = glt(q) * Li;
                    for (size_t r = 0; r < n_qp; r++) {
                        double wj = glw(r) * Lj;
                        double uj = glt(r) * Lj;
                        double Rv = R[q * n_qp + r];
                        // exp(-j k R) - 1 = (cos(kR) - 1) - j sin(kR)
                        //   = (cos(-kR) - 1) + j sin(-kR)
                        double phase = -k * Rv;
                        double cm1   = std::cos(phase) - 1.0;
                        double sp    = std::sin(phase);
                        std::complex<double> Greg(cm1, sp);
                        Greg *= inv_4pi / Rv;
                        std::complex<double> wG = (wi * wj) * Greg;
                        s00 += wG;
                        s10 += ui * wG;
                        s01 += uj * wG;
                        s11 += (ui * uj) * wG;
                    }
                }
                j00(kk, i, j) = s00;
                j10(kk, i, j) = s10;
                j01(kk, i, j) = s01;
                j11(kk, i, j) = s11;
            }
        }
    }

    return std::make_tuple(J00, J10, J01, J11);
}


std::complex<double> trapezoid_aux(double theta, double delta, double *m_center_ptr, double *n_l_endpoint_ptr, double *n_r_endpoint_ptr, double wire_radius, double k) {

  std::complex<double> minus_jk = -1i*k;

  double R;
  {
    double sumsq = 0.0;
    for (size_t kk=0; kk<3; ++kk) {
      auto diff = n_l_endpoint_ptr[kk]*(1-theta) + n_r_endpoint_ptr[kk]*theta - m_center_ptr[kk];
      sumsq += diff*diff;
    }
    R = sqrt(sumsq);
  }

  std::complex<double> res;
  if (R < 0.00001) {
    res = 1.0/(2.0*M_PI*delta) * log(delta/wire_radius) + minus_jk/(4.0*M_PI);
  } else {
    res = exp(minus_jk*R)/(4*M_PI*R);
  }

  return res;
}


py::array_t<std::complex<double> > psi_fusion_trapezoid(
    py::array_t<double> input0,
    py::array_t<double> input1,
    double wire_radius,
    double k,
    int ntrap
) {
  auto buf0 = input0.request();
  auto buf1 = input1.request();

  if (buf0.ndim != 2)
    throw std::runtime_error("Number of dimensions must be two");

  if (buf1.ndim != 2)
    throw std::runtime_error("Number of dimensions must be two");

  if (buf0.shape[0] % 2 != 1)
    throw std::runtime_error("Input0 must have odd first dimension size");

  if (buf1.shape[0] % 2 != 1)
    throw std::runtime_error("Input1 must have odd first dimension size");

  if (buf0.shape[1] != buf1.shape[1])
    throw std::runtime_error("Inputs must have same sized second dimension");

  size_t rows = (buf0.shape[0]-1)/2;
  size_t cols = (buf1.shape[0]-1)/2;
  size_t vsize = buf0.shape[1];

  auto result = py::array_t<std::complex<double> >({rows, cols});
  auto result_buf = result.request();
  
  double *ptr0 = static_cast<double *>(buf0.ptr);
  double *ptr1 = static_cast<double *>(buf1.ptr);

  std::complex<double> *result_ptr = static_cast<std::complex<double> *>(result_buf.ptr);

  double one_over_ntrap = 1.0/ntrap;
  double one_over_2_ntrap = 0.5*one_over_ntrap;

  #pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    auto m_center_ptr = ptr0 + (2*i+1)*vsize;

    for (size_t j = 0; j < cols; j++) {

      auto n_l_endpoint_ptr = ptr1 + (2*(j+0))*vsize;
      auto n_r_endpoint_ptr = ptr1 + (2*(j+1))*vsize;

      double delta;
      {
	double sumsq = 0.0;
	for (size_t kk=0; kk<3; ++kk) {
	  auto diff = n_r_endpoint_ptr[kk] - n_l_endpoint_ptr[kk];
	  sumsq += diff*diff;
	}
	delta = sqrt(sumsq);
      }

      std::complex<double> res = 0.0;

      if (ntrap == 0) {
         res = trapezoid_aux(0.5, delta, m_center_ptr, n_l_endpoint_ptr, n_r_endpoint_ptr, wire_radius, k);
      } else {
	for(size_t kk=0; kk<ntrap+1; kk++) {
	  double theta = static_cast<double>(kk)*one_over_ntrap;
	  double coeff = one_over_2_ntrap;
	  if (kk>0 && kk<ntrap) {
	    coeff = one_over_ntrap;
	  }
	  res += coeff*trapezoid_aux(theta, delta, m_center_ptr, n_l_endpoint_ptr, n_r_endpoint_ptr, wire_radius, k);
	}
      }
      result_ptr[i*cols+j] = res;
    }
  }


  return result;
}


py::array_t<double> dist_outer_product(py::array_t<double> input0,
				       py::array_t<double> input1) {
    auto buf0 = input0.request();
    auto buf1 = input1.request();

    if (buf0.ndim != 2)
      throw std::runtime_error("Number of dimensions must be two");

    if (buf1.ndim != 2)
      throw std::runtime_error("Number of dimensions must be two");

    if (buf0.shape[1] != buf1.shape[1])
      throw std::runtime_error("Inputs must have same sized second dimension");
    
    size_t rows = buf0.shape[0];
    size_t cols = buf1.shape[0];
    size_t vsize = buf0.shape[1];

    auto result = py::array_t<double>({rows, cols});
    auto result_buf = result.request();

    double *ptr0 = static_cast<double *>(buf0.ptr);
    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *result_ptr = static_cast<double *>(result_buf.ptr);

    #pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
	auto sumsq = 0.0;
	for (size_t k = 0; k < vsize; k++) {
	  auto diff = ptr0[i*vsize+k] - ptr1[j*vsize+k];
	  sumsq += diff*diff;
	}
        result_ptr[i*cols+j] = sqrt(sumsq);
      }
    }

    return result;
}

PYBIND11_MODULE(_accelerators, m) {
    m.def("dist_outer_product", &dist_outer_product, "Compute point to point euclidean distance");
    m.def("psi_fusion_trapezoid", &psi_fusion_trapezoid, "Compute Psi (Integral) from point vectors using trapezoidal method", py::arg("input0"), py::arg("input1"), py::kw_only(), py::arg("wire_radius"), py::arg("k"), py::arg("ntrap"));
    m.def("seg_seg_quad_batch_3d", &seg_seg_quad_batch_3d,
          "Batched 3D cross-segment Gauss-Legendre quadrature over a k vector. "
          "Returns (J00, J10, J01, J11) each (n_k, N_i, N_j) complex.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("seg_seg_reg_quad_batch_1d", &seg_seg_reg_quad_batch_1d,
          "Batched same-wire Gauss-Legendre quadrature on the regularized "
          "kernel (exp(-jkR)-1)/(4 pi R) over a k vector. "
          "Returns (J00, J10, J01, J11) each (n_k, N, N) complex.",
          py::arg("seg_endpoints"), py::arg("a"),
          py::arg("k_array"),
          py::arg("gl_t"), py::arg("gl_w"));
}
