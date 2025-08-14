#include "Optimizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <iomanip>
#include <limits>
#include <sstream>

using Vec3 = Eigen::Vector3d;

Optimizer::Optimizer(const OptimizerOptions& opt) : opt_(opt) {}

static inline void unpackA_t_rowmajor(const Eigen::VectorXd& x, int node_id, Eigen::Matrix3d& A, Vec3& t) {
    const int b = 12*node_id;
    // row-major: [a11 a12 a13 a21 a22 a23 a31 a32 a33]
    A << x(b+0),x(b+1),x(b+2),
         x(b+3),x(b+4),x(b+5),
         x(b+6),x(b+7),x(b+8);
    t = Vec3(x(b+9), x(b+10), x(b+11));
}

static inline void addBlock3x9_A_rowmajor(Eigen::MatrixXd& J, int r0, int c0, const Vec3& v) {
    // Derivative of A*v w.r.t row-major A entries: block = [v_x I3 | v_y I3 | v_z I3] arranged by rows
    // Column order in variables: [a11 a12 a13 | a21 a22 a23 | a31 a32 a33]
    // For each output row p (0..2): d(A*v)_p/ d a_{p,1} = v_x, d a_{p,2} = v_y, d a_{p,3} = v_z
    // So block is:
    // [ v.x  v.y  v.z   0    0    0    0    0    0 ] for row 0 but placed at columns corresponding to a11,a12,a13 of row0, etc.
    // Easier: fill 3x9 as below
    J.block<3,9>(r0, c0).setZero();
    // row 0 affects a11,a12,a13 (cols c0+0..2)
    J(r0+0, c0+0) = v.x(); J(r0+0, c0+1) = v.y(); J(r0+0, c0+2) = v.z();
    // row 1 affects a21,a22,a23 (cols c0+3..5)
    J(r0+1, c0+3) = v.x(); J(r0+1, c0+4) = v.y(); J(r0+1, c0+5) = v.z();
    // row 2 affects a31,a32,a33 (cols c0+6..8)
    J(r0+2, c0+6) = v.x(); J(r0+2, c0+7) = v.y(); J(r0+2, c0+8) = v.z();
}

void Optimizer::buildResidualVector(const Eigen::VectorXd& x,
                                    const EDGraph& ed,
                                    const std::vector<Vec3>& key_old,
                                    const std::vector<Vec3>& key_new,
                                    const std::vector<int>& key_idx,
                                    Eigen::VectorXd& F,
                                    Eigen::VectorXd& v_diag,
                                    int& num_rownode,
                                    int& G,
                                    int& Kc) const
{
    G  = ed.numNodes();
    const auto& nodes = ed.getGraphNodes();
    const auto& neigh = ed.getNodeNeighbors();

    const int num_nearestpts = (int)neigh.empty() ? 1 : ((int)neigh[0].size() + 1);
    num_rownode = 6 + 3 * num_nearestpts; // for MATLAB-style layout only

    Kc = (int)key_old.size();

    int total_rows = 0;
    for (int i = 0; i < G; ++i) total_rows += 6 + 3 * (int)neigh[i].size();
    total_rows += 3 * Kc;

    F.resize(total_rows);
    v_diag.resize(total_rows);

    int row = 0;
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix3d A; Vec3 ti; unpackA_t_rowmajor(x, i, A, ti);
        const Vec3 gi = nodes[i].position;
        const Vec3 c0 = A.col(0), c1 = A.col(1), c2 = A.col(2);
        // 6 orthogonality residuals
        F[row+0] = c0.dot(c1);
        F[row+1] = c0.dot(c2);
        F[row+2] = c1.dot(c2);
        F[row+3] = c0.squaredNorm() - 1.0;
        F[row+4] = c1.squaredNorm() - 1.0;
        F[row+5] = c2.squaredNorm() - 1.0;
        for (int k=0;k<6;++k) v_diag[row+k] = opt_.w_rot_rows;
        row += 6;
        // smoothness residuals (i->j, double-count)
        for (int j : neigh[i]) {
            Eigen::Matrix3d Aj; Vec3 tj; unpackA_t_rowmajor(x, j, Aj, tj);
            const Vec3 gj = nodes[j].position;
            Vec3 rij = A * (gj - gi) + gi + ti - (gj + tj);
            F.segment<3>(row) = rij;
            v_diag.segment<3>(row).setConstant(opt_.w_conn_rows);
            row += 3;
        }
    }

    // data term: use pre-bound vertex weights (same as current pipeline)
    const auto& B = ed.getVertexBindings();
    const auto& W = ed.getVertexWeights();
    for (int k = 0; k < Kc; ++k) {
        const int vid = key_idx[k];
        Vec3 pred = ed.deformVertexByState(key_old[k], x, B[vid], W[vid], 0);
        Vec3 r = pred - key_new[k];
        F.segment<3>(row) = r;
        v_diag.segment<3>(row).setConstant(opt_.w_data_rows);
        row += 3;
    }
}

// Analytic Jacobian that mirrors MATLAB JacobianF (with 0.6 factors in smoothness)
void Optimizer::analyticJacobian(const Eigen::VectorXd& x,
                                 const EDGraph& ed,
                                 const std::vector<Vec3>& key_old,
                                 const std::vector<int>& key_idx,
                                 Eigen::MatrixXd& J) const
{
    const int G  = ed.numNodes();
    const auto& nodes = ed.getGraphNodes();
    const auto& neigh = ed.getNodeNeighbors();

    // Count rows exactly like buildResidualVector
    int total_rows = 0;
    for (int i = 0; i < G; ++i) total_rows += 6 + 3 * (int)neigh[i].size();
    total_rows += 3 * (int)key_old.size();

    J.setZero(total_rows, 12*G);

    // --- Orthonormality blocks (per node, 6 rows) ---
    int row = 0;
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix3d A; Vec3 ti; unpackA_t_rowmajor(x, i, A, ti);
        // Columns (c0,c1,c2)
        const Eigen::Vector3d c0 = A.col(0), c1 = A.col(1), c2 = A.col(2);
        const int base = 12*i; // variables start index

        // Helpers: indices in row-major
        auto idxA = [&](int r, int c){ return base + (r*3 + c); }; // r,c in [0..2]
        // d/dt = 0 for ortho

        // r1 = c0·c1 -> derivative wrt a_{k1} = c1_k; wrt a_{k2} = c0_k
        for (int k=0;k<3;++k) {
            J(row+0, idxA(k,0)) = c1[k]; // a_{k1}
            J(row+0, idxA(k,1)) = c0[k]; // a_{k2}
        }
        // r2 = c0·c2
        for (int k=0;k<3;++k) {
            J(row+1, idxA(k,0)) = c2[k];
            J(row+1, idxA(k,2)) = c0[k];
        }
        // r3 = c1·c2
        for (int k=0;k<3;++k) {
            J(row+2, idxA(k,1)) = c2[k];
            J(row+2, idxA(k,2)) = c1[k];
        }
        // r4 = ||c0||^2 - 1
        for (int k=0;k<3;++k) J(row+3, idxA(k,0)) = 2.0 * c0[k];
        // r5 = ||c1||^2 - 1
        for (int k=0;k<3;++k) J(row+4, idxA(k,1)) = 2.0 * c1[k];
        // r6 = ||c2||^2 - 1
        for (int k=0;k<3;++k) J(row+5, idxA(k,2)) = 2.0 * c2[k];

        row += 6;

        // --- Smoothness edges: for each neighbor j ---
        const Vec3 gi = nodes[i].position;
        for (int j : neigh[i]) {
            const int base_i = 12*i;
            const int base_j = 12*j;
            const Vec3 gj = nodes[j].position;
            const Vec3 dgj = gj - gi; // Δ = g_j - g_i

            // w.r.t A_i: d/dA_i (A_i * Δ) = [Δ_x I, Δ_y I, Δ_z I] in row-major order
            addBlock3x9_A_rowmajor(J, row, base_i + 0, dgj);

            // w.r.t t_i: +0.6 * I3  (MATLAB JacobianF has 0.6 here)
            J.block<3,3>(row, base_i + 9).setIdentity();
            J.block<3,3>(row, base_i + 9) *= 0.6;

            // w.r.t t_j: -0.6 * I3
            J.block<3,3>(row, base_j + 9).setIdentity();
            J.block<3,3>(row, base_j + 9) *= -0.6;

            row += 3;
        }
    }

    // --- Data term: r_k = Σ w_kℓ [ A_ℓ (p - g_ℓ) + g_ℓ + t_ℓ ] - target  ---
    const auto& B = ed.getVertexBindings();
    const auto& W = ed.getVertexWeights();
    const int Kc = (int)key_old.size();

    for (int k = 0; k < Kc; ++k) {
        const int vid = key_idx[k];
        const auto& nodes_idx = B[vid];          // indices ℓ
        const auto& weights   = W[vid];          // weights w_kℓ
        // exploit storage: assume nodes_idx.size()==weights.size()==K_bind
        for (size_t q = 0; q < nodes_idx.size(); ++q) {
            const int l = nodes_idx[q];
            const double w = weights[q];
            const int base_l = 12*l;
            const Vec3 gl = nodes[l].position;
            Vec3 d = key_old[k] - gl; // (p - g_l)
            // d/dA_l : w * [d_x I, d_y I, d_z I] (row-major)
            addBlock3x9_A_rowmajor(J, row, base_l + 0, w*d);
            // d/dt_l : w * I3
            J.block<3,3>(row, base_l + 9).noalias() += w * Eigen::Matrix3d::Identity();
        }
        row += 3;
    }
}

namespace {
struct RawBreakdown {
    double ortho = 0.0, smooth = 0.0, data = 0.0, cost_raw = 0.0, cost_P_half = 0.0;
    double key_mean = std::numeric_limits<double>::quiet_NaN();
    double key_rmse = std::numeric_limits<double>::quiet_NaN();
    double key_max  = std::numeric_limits<double>::quiet_NaN();
};

static RawBreakdown compute_breakdown(const Eigen::VectorXd& F,
                                      const Eigen::VectorXd& v_diag,
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
        if (srows > 0) { R.smooth += 0.5 * F.segment(row, srows).squaredNorm(); row += srows; }
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
static std::string fmt_g6(double v){ std::ostringstream ss; ss.setf(std::ios::fmtflags(0), std::ios::floatfield); ss<<std::setprecision(6)<<v; return ss.str(); }
static std::string fmt_e3(double v){ std::ostringstream ss; ss<<std::scientific<<std::setprecision(3)<<v; return ss.str(); }
}

void Optimizer::optimize(EDGraph& edgraph,
                         Eigen::VectorXd& x,
                         const std::vector<Vec3>& key_old,
                         const std::vector<Vec3>& key_new,
                         const std::vector<int>&   key_indices)
{
    auto buildF = [&](const Eigen::VectorXd& xt, Eigen::VectorXd& Fout, Eigen::VectorXd& vdiag){
        int nrn, G, Kc; buildResidualVector(xt, edgraph, key_old, key_new, key_indices, Fout, vdiag, nrn, G, Kc);
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

    // Analytic J at x
    Eigen::MatrixXd J; analyticJacobian(x, edgraph, key_old, key_indices, J);

    auto assemble = [&](const Eigen::MatrixXd& Jm, const Eigen::VectorXd& Fm, const Eigen::VectorXd& vdiag){
        Eigen::VectorXd inv_v = vdiag.cwiseInverse();
        Eigen::MatrixXd JP = Jm.transpose() * inv_v.asDiagonal();
        Eigen::MatrixXd H  = JP * Jm;
        Eigen::VectorXd g  = JP * Fm;
        double cost = phi(Fm, vdiag);
        return std::make_tuple(H, g, cost);
    };

    Eigen::MatrixXd H; Eigen::VectorXd g; double cost; std::tie(H,g,cost) = assemble(J,F,v_diag);

    double prev_cost = 1e300;
    int it = 0;
    for (; it < opt_.max_iters; ++it) {
        // Gauss-Newton direction
        Eigen::VectorXd d = H.ldlt().solve(-g);

        // Line search (MATLAB-like)
        const double phi0 = cost; const double phi0_deriv = d.dot(g);
        double alpha = opt_.alpha0, step = opt_.step0;

        auto eval_at = [&](double a){
            Eigen::VectorXd xt = x + a * d;
            Eigen::VectorXd Ft, Pt; buildF(xt, Ft, Pt);
            Eigen::MatrixXd Jt; analyticJacobian(xt, edgraph, key_old, key_indices, Jt);
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

        // Accept step
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

        // Reassemble H,g at new x
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