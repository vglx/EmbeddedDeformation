#include "Optimizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <iomanip>
#include <limits>

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
    // Note: per node rows actually = 6 + 3*neigh[i].size(). num_rownode is for logging layout only
    num_rownode = 6 + 3 * num_nearestpts; // mirrors MATLAB intent (K includes self)

    Kc = (int)key_old.size();

    // Compute total rows: rotation (6 per node) + smoothness (3*|neigh[i]| per node) + data (3*Kc)
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
                                      const Eigen::VectorXd& v_diag, // Pdiag here stores v (not its inverse)
                                      const EDGraph& ed)
{
    RawBreakdown R;
    const auto& neigh = ed.getNodeNeighbors();
    const int G = (int)neigh.size();

    // Sum per-node: 6 ortho + 3*|neigh[i]| smooth
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

    // data part
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
    // weighted cost: 0.5 * sum_i F_i^2 / v_i (since v_diag stores MATLAB v)
    R.cost_P_half = 0.5 * F.cwiseQuotient(v_diag).dot(F);
    return R;
}
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
    auto phi = [&](const Eigen::VectorXd& F, const Eigen::VectorXd& vdiag){
        // cost = F' * inv(diag(v_diag)) * F
        return F.cwiseQuotient(vdiag).dot(F);
    };

    // initial F, J, H, g, cost
    Eigen::VectorXd F, v_diag; buildF(x, F, v_diag);

    // MATLAB-style init logs
    if (opt_.verbose) {
        auto R0 = compute_breakdown(F, v_diag, edgraph);
        std::cout << std::setprecision(7)
                  << "[Init]   key_mean=" << R0.key_mean
                  << "  key_rmse=" << R0.key_rmse
                  << "  key_max="  << R0.key_max << "\n";
        std::cout << std::setprecision(12)
                  << "[GN it=0] cost(raw)=" << R0.cost_raw
                  << "  (data=" << R0.data << ", smooth=" << R0.smooth << ", ortho=" << R0.ortho << ")"
                  << "  |dx|=n/a  ||F||_P^2/2=" << R0.cost_P_half << "\n";
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
    if (opt_.verbose) std::cout << "[GN] init cost=" << std::setprecision(12) << (0.5*cost)
                                 << "  n=" << x.size() << "  m=" << F.size() << "\n";

    double prev_cost = 1e300;
    int it = 0;
    for (; it < opt_.max_iters; ++it) {
        // Gauss-Newton direction: d = -(J' P J)^{-1} J' P F
        Eigen::VectorXd d = H.ldlt().solve(-g);

        // Line search (MATLAB-like)
        const double phi0 = cost;
        const double phi0_deriv = d.dot(g); // since g = J' P F
        double alpha = opt_.alpha0;
        double step  = opt_.step0;

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

        if (!ok) {
            if (opt_.verbose) std::cout << "[GN] it=" << it << "  line-search failed, alpha→0\n";
            break;
        }

        // Accept step
        x += alpha * d;
        F.swap(Fa); v_diag.swap(Pa); J.swap(Ja); cost = phia;

        // MATLAB-style per-iteration print (use it+1 to match script's first update)
        if (opt_.verbose) {
            auto R = compute_breakdown(F, v_diag, edgraph);
            std::cout << std::setprecision(8)
                      << "[GN it=" << (it+1) << "] cost(raw)=" << R.cost_raw
                      << "  (data=" << R.data << ", smooth=" << R.smooth << ", ortho=" << R.ortho << ")"
                      << "  |dx|=" << d.norm()
                      << "  ||F||_P^2/2=" << std::setprecision(12) << R.cost_P_half
                      << "  [key mean=" << std::setprecision(8) << R.key_mean
                      << " rmse=" << R.key_rmse
                      << " max=" << R.key_max << "]\n";
        }

        // Convergence checks (match MATLAB spirit)
        if (0.5*cost < opt_.tol_cost) break;
        if (std::abs(prev_cost - cost) < opt_.tol_cost) break;
        prev_cost = cost;

        // Recompute H,g at new x
        {
            Eigen::VectorXd inv_v = v_diag.cwiseInverse();
            Eigen::MatrixXd JP = J.transpose() * inv_v.asDiagonal();
            H  = JP * J;
            g  = JP * F;
        }
    }

    if (opt_.verbose) {
        auto Rf = compute_breakdown(F, v_diag, edgraph);
        std::cout << "[GN] finished iters=" << it << "  final cost=" << std::setprecision(12) << (0.5*cost) << "\n";
        std::cout << std::setprecision(8)
                  << "[Final]  key_mean=" << Rf.key_mean
                  << "  key_rmse=" << Rf.key_rmse
                  << "  key_max="  << Rf.key_max << "\n";
    }

    edgraph.updateFromStateVector(x, 0);
}