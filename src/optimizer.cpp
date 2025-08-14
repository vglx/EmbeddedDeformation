#include "Optimizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <iomanip>
#include <limits>
#include <sstream>

using Vec3 = Eigen::Vector3d;

Optimizer::Optimizer(const OptimizerOptions& opt) : opt_(opt) {}

static inline void unpackA_t(const Eigen::VectorXd& x, int node_id, Eigen::Matrix3d& A, Vec3& t) {
    const int b = 12*node_id;
    A << x(b+0),x(b+1),x(b+2),
         x(b+3),x(b+4),x(b+5),
         x(b+6),x(b+7),x(b+8);
    t = Vec3(x(b+9), x(b+10), x(b+11));
}

void Optimizer::buildResidualVector(const Eigen::VectorXd& x,
                                    const EDGraph& ed,
                                    const std::vector<Vec3>& key_old,
                                    const std::vector<Vec3>& key_new,
                                    const std::vector<int>& key_idx,
                                    Eigen::VectorXd& F,
                                    Eigen::VectorXd& Pdiag,
                                    int& num_rownode,
                                    int& G,
                                    int& Kc) const
{
    G  = ed.numNodes();
    const auto& nodes = ed.getGraphNodes();
    const auto& neigh = ed.getNodeNeighbors();

    // Deduce K from neighbors: size(neigh[i]) = num_nearestpts-1 typically
    const int num_nearestpts = (int)neigh.empty() ? 1 : ((int)neigh[0].size() + 1);
    // Note: per node rows actually = 6 + 3*neigh[i].size(). num_rownode is only for "layout" like MATLAB
    num_rownode = 6 + 3 * num_nearestpts; // mirrors MATLAB intent (K includes self)

    Kc = (int)key_old.size();

    // total rows
    int total_rows = 0;
    for (int i = 0; i < G; ++i) total_rows += 6 + 3 * (int)neigh[i].size();
    total_rows += 3 * Kc;

    F.resize(total_rows);
    Pdiag.resize(total_rows);

    int row = 0;
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix3d A; Vec3 ti; unpackA_t(x, i, A, ti);
        const Vec3 gi = nodes[i].position;
        // rotation residuals: 6 rows
        const Vec3 c0 = A.col(0), c1 = A.col(1), c2 = A.col(2);
        F[row+0] = c0.dot(c1);
        F[row+1] = c0.dot(c2);
        F[row+2] = c1.dot(c2);
        F[row+3] = c0.squaredNorm() - 1.0;
        F[row+4] = c1.squaredNorm() - 1.0;
        F[row+5] = c2.squaredNorm() - 1.0;
        for (int k=0;k<6;++k) Pdiag[row+k] = opt_.w_rot_rows;
        row += 6;
        // smoothness residuals: double-count i->j as per MATLAB traversal
        for (int j : neigh[i]) {
            Eigen::Matrix3d Aj; Vec3 tj; unpackA_t(x, j, Aj, tj);
            const Vec3 gj = nodes[j].position;
            Vec3 rij = A * (gj - gi) + gi + ti - (gj + tj);
            F.segment<3>(row) = rij;
            Pdiag.segment<3>(row).setConstant(opt_.w_conn_rows);
            row += 3;
        }
    }

    // data term
    const auto& B = ed.getVertexBindings();
    const auto& W = ed.getVertexWeights();
    for (int k = 0; k < Kc; ++k) {
        const int vid = key_idx[k];
        Vec3 pred = ed.deformVertexByState(key_old[k], x, B[vid], W[vid], 0);
        Vec3 r = pred - key_new[k];
        F.segment<3>(row) = r;
        Pdiag.segment<3>(row).setConstant(opt_.w_data_rows);
        row += 3;
    }
}

void Optimizer::numericJacobian(std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> f,
                                const Eigen::VectorXd& x,
                                Eigen::MatrixXd& J,
                                double eps) const
{
    Eigen::VectorXd f0; f(x, f0);
    const int m = (int)f0.size();
    const int n = (int)x.size();
    J.resize(m, n);

    Eigen::VectorXd xt = x;
    for (int j = 0; j < n; ++j) {
        const double h = eps * std::max(1.0, std::abs(x[j]));
        xt[j] = x[j] + h; Eigen::VectorXd fp; f(xt, fp);
        xt[j] = x[j] - h; Eigen::VectorXd fm; f(xt, fm);
        xt[j] = x[j];
        J.col(j) = (fp - fm) / (2.0*h);
    }
}

namespace {
struct RawBreakdown {
    double ortho = 0.0, smooth = 0.0, data = 0.0, cost_raw = 0.0, cost_P_half = 0.0;
    double key_mean = std::numeric_limits<double>::quiet_NaN();
    double key_rmse = std::numeric_limits<double>::quiet_NaN();
    double key_max  = std::numeric_limits<double>::quiet_NaN();
};

// Compute MATLAB-style raw-cost breakdown and weighted cost/2
static RawBreakdown compute_breakdown(const Eigen::VectorXd& F,
                                      const Eigen::VectorXd& v_diag, // v (not inverse)
                                      const EDGraph& ed)
{
    RawBreakdown R;
    const auto& neigh = ed.getNodeNeighbors();
    const int G = (int)neigh.size();

    int row = 0;
    for (int i = 0; i < G; ++i) {
        R.ortho  += 0.5 * F.segment<6>(row).squaredNorm();
        row += 6;
        const int srows = 3 * (int)neigh[i].size();
        if (srows > 0) {
            R.smooth += 0.5 * F.segment(row, srows).squaredNorm();
            row += srows;
        }
    }

    const int len_data = (int)F.size() - row;
    if (len_data > 0) {
        R.data = 0.5 * F.tail(len_data).squaredNorm();
        const int Kc = len_data / 3;
        if (Kc > 0) {
            Eigen::Map<const Eigen::Matrix<double,3,Eigen::Dynamic>> RD(F.data()+row, 3, Kc);
            Eigen::ArrayXd norms = RD.colwise().norm().transpose().array();
            R.key_mean = norms.mean();
            R.key_rmse = std::sqrt(norms.square().mean());
            R.key_max  = norms.maxCoeff();
        }
    }

    R.cost_raw   = R.ortho + R.smooth + R.data;
    R.cost_P_half = 0.5 * F.cwiseQuotient(v_diag).dot(F); // 0.5 * sum F_i^2 / v_i
    return R;
}

// helpers to mimic MATLAB's %.6g / %.3e formatting regardless of global iostream flags
static std::string fmt_g6(double v){ std::ostringstream ss; ss.setf(std::ios::fmtflags(0), std::ios::floatfield); ss<<std::setprecision(6)<<v; return ss.str(); }
static std::string fmt_e3(double v){ std::ostringstream ss; ss<<std::scientific<<std::setprecision(3)<<v; return ss.str(); }
}

void Optimizer::optimize(EDGraph& edgraph,
                         Eigen::VectorXd& x,
                         const std::vector<Vec3>& key_old,
                         const std::vector<Vec3>& key_new,
                         const std::vector<int>&   key_indices)
{
    auto buildF = [&](const Eigen::VectorXd& xt, Eigen::VectorXd& Fout, Eigen::VectorXd& Pdiag){
        int nrn, G, Kc; buildResidualVector(xt, edgraph, key_old, key_new, key_indices, Fout, Pdiag, nrn, G, Kc);
    };
    auto phi = [&](const Eigen::VectorXd& F, const Eigen::VectorXd& vdiag){ return F.cwiseQuotient(vdiag).dot(F); };

    Eigen::VectorXd F, v_diag; buildF(x, F, v_diag);

    if (opt_.verbose) {
        auto R0 = compute_breakdown(F, v_diag, edgraph);
        std::cout << "[Init]   key_mean=" << fmt_g6(R0.key_mean)
                  << "  key_rmse=" << fmt_g6(R0.key_rmse)
                  << "  key_max="  << fmt_g6(R0.key_max) << "\n";
        std::cout << "[GN it=0] cost(raw)=" << fmt_g6(R0.cost_raw)
                  << "  (data=" << fmt_g6(R0.data) << ", smooth=" << fmt_g6(R0.smooth) << ", ortho=" << fmt_g6(R0.ortho) << ")"
                  << "  |dx|=n/a  ||F||_P^2/2=" << fmt_g6(R0.cost_P_half) << "\n";
    }

    auto Ffun = [&](const Eigen::VectorXd& xt, Eigen::VectorXd& Fout){ Eigen::VectorXd P; buildF(xt, Fout, P); };
    Eigen::MatrixXd J; numericJacobian(Ffun, x, J);

    auto assemble = [&](const Eigen::MatrixXd& Jm, const Eigen::VectorXd& Fm, const Eigen::VectorXd& vdiag){
        Eigen::VectorXd inv_v = vdiag.cwiseInverse();
        Eigen::MatrixXd JP = Jm.transpose() * inv_v.asDiagonal();
        Eigen::MatrixXd H  = JP * Jm;
        Eigen::VectorXd g  = JP * Fm;
        double cost = phi(Fm, vdiag);
        return std::make_tuple(H, g, cost);
    };

    Eigen::MatrixXd H; Eigen::VectorXd g; double cost; std::tie(H,g,cost) = assemble(J,F,v_diag);
    // 移除 "[GN] init cost=..." 这一行，完全用 MATLAB 风格

    double prev_cost = 1e300;
    int it = 0;
    for (; it < opt_.max_iters; ++it) {
        Eigen::VectorXd d = H.ldlt().solve(-g);

        const double phi0 = cost; const double phi0_deriv = d.dot(g);
        double alpha = opt_.alpha0, step = opt_.step0;

        auto eval_at = [&](double a){
            Eigen::VectorXd xt = x + a * d;
            Eigen::VectorXd Ft, Pt; buildF(xt, Ft, Pt);
            Eigen::MatrixXd Jt; numericJacobian(Ffun, xt, Jt);
            double phit = phi(Ft, Pt);
            Eigen::VectorXd inv_v = Pt.cwiseInverse();
            Eigen::VectorXd gt = Jt.transpose() * (inv_v.asDiagonal() * Ft);
            double dphit = d.dot(gt);
            return std::tuple<double,double,Eigen::VectorXd,Eigen::VectorXd,Eigen::MatrixXd>(phit, dphit, Ft, Pt, Jt);
        };

        int k = 0; bool ok = false; double phia=0.0, dphia=0.0; Eigen::VectorXd Fa, Pa; Eigen::MatrixXd Ja;
        while (k < 10) {
            double phit, dphit; Eigen::VectorXd Ft, Pt; Eigen::MatrixXd Jt;
            std::tie(phit, dphit, Ft, Pt, Jt) = eval_at(alpha);
            bool cond1 = (phit <= phi0 + opt_.gamma1 * phi0_deriv * alpha);
            bool cond2 = (dphit >= opt_.gamma2 * phi0_deriv);
            if (cond1 && cond2) { ok = true; phia = phit; dphia = dphit; Fa = Ft; Pa = Pt; Ja = Jt; break; }
            if (!cond1) { alpha -= step; step *= 0.5; }
            else if (!cond2) { alpha += step; step *= 0.5; }
            if (alpha <= 1e-12) break;
            ++k;
        }
        if (!ok) { if (opt_.verbose) std::cout << "[GN it=" << it << "] line-search failed, alpha->0\n"; break; }

        x += alpha * d; F.swap(Fa); v_diag.swap(Pa); J.swap(Ja); cost = phia;

        if (opt_.verbose) {
            auto R = compute_breakdown(F, v_diag, edgraph);
            std::cout << "[GN it=" << (it+1) << "] cost(raw)=" << fmt_g6(R.cost_raw)
                      << "  (data=" << fmt_g6(R.data) << ", smooth=" << fmt_g6(R.smooth) << ", ortho=" << fmt_g6(R.ortho) << ")"
                      << "  |dx|=" << fmt_e3(d.norm())
                      << "  ||F||_P^2/2=" << fmt_g6(R.cost_P_half)
                      << "  [key mean=" << fmt_g6(R.key_mean)
                      << " rmse=" << fmt_g6(R.key_rmse)
                      << " max=" << fmt_g6(R.key_max) << "]\n";
        }

        if (0.5*cost < opt_.tol_cost) break;
        if (std::abs(prev_cost - cost) < opt_.tol_cost) break;
        prev_cost = cost;

        Eigen::VectorXd inv_v = v_diag.cwiseInverse();
        Eigen::MatrixXd JP = J.transpose() * inv_v.asDiagonal();
        H  = JP * J; g  = JP * F;
    }

    if (opt_.verbose) {
        auto Rf = compute_breakdown(F, v_diag, edgraph);
        std::cout << "[GN] finished iters=" << it << "  final cost=" << fmt_g6(0.5*cost) << "\n";
        std::cout << "[Final]  key_mean=" << fmt_g6(Rf.key_mean)
                  << "  key_rmse=" << fmt_g6(Rf.key_rmse)
                  << "  key_max="  << fmt_g6(Rf.key_max) << "\n";
    }

    edgraph.updateFromStateVector(x, 0);
}