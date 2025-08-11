#include "Optimizer.h"
#include <iostream>
#include <numeric>

using Vec3 = Eigen::Vector3d;

// Build keypoint (data) residuals: r_data = w_data * (ED_x(key_old) - key_new)
static void eval_residual_data(const Eigen::VectorXd& x,
                               const EDGraph& edgraph,
                               const std::vector<Vec3>& key_old,
                               const std::vector<Vec3>& key_new,
                               const std::vector<int>& key_indices,
                               Eigen::VectorXd& r_data,
                               double w_data)
{
    const auto& B = edgraph.getBindings();
    const auto& W = edgraph.getWeights();
    const int Nk = static_cast<int>(key_old.size());
    r_data.resize(3 * Nk);

    for (int i = 0; i < Nk; ++i) {
        const int vid = key_indices[i];
        const auto& ids = B[vid];
        std::vector<double> ws = W[vid];
        double s = 0.0; for (double v : ws) s += v; if (s>0) for (double& v : ws) v /= s;
        Vec3 pred = edgraph.deformVertexByState(key_old[i], x, ids, ws, /*offset=*/0);
        r_data.segment<3>(3*i) = w_data * (pred - key_new[i]);
    }
}

// Build smoothness residuals over node graph: for each undirected edge (i<j)
// r_ij = T_i*(g_j - g_i) + g_i + t_i - (g_j + t_j)
static void eval_residual_smooth(const Eigen::VectorXd& x,
                                 const EDGraph& edgraph,
                                 Eigen::VectorXd& r_smooth,
                                 double w_smooth)
{
    const int G = edgraph.numNodes();
    const auto& nodes = edgraph.getGraphNodes();
    const auto& neigh = edgraph.getNodeNeighbors();

    // Count undirected edges (i<j)
    int E = 0;
    for (int i = 0; i < G; ++i) {
        for (int j : neigh[i]) if (j > i) ++E;
    }
    r_smooth.resize(3 * E);

    int e = 0;
    for (int i = 0; i < G; ++i) {
        // Decode Ti from x
        Eigen::Matrix<double,6,1> se3_i = x.segment<6>(6 * i);
        Sophus::SE3d Ti = Sophus::SE3d::exp(se3_i);
        const Vec3 gi = nodes[i].position;

        for (int j : neigh[i]) if (j > i) {
            Eigen::Matrix<double,6,1> se3_j = x.segment<6>(6 * j);
            Sophus::SE3d Tj = Sophus::SE3d::exp(se3_j);
            const Vec3 gj = nodes[j].position;

            Vec3 lhs = Ti.so3() * (gj - gi) + gi + Ti.translation();
            Vec3 rhs = gj + Tj.translation();
            r_smooth.segment<3>(3*e) = w_smooth * (lhs - rhs);
            ++e;
        }
    }
}

// Build full residual r = [r_data; r_smooth]; also return split costs for logging
static void eval_residuals_full(const Eigen::VectorXd& x,
                                const EDGraph& edgraph,
                                const std::vector<Vec3>& key_old,
                                const std::vector<Vec3>& key_new,
                                const std::vector<int>& key_indices,
                                Eigen::VectorXd& r,
                                double& cost_data,
                                double& cost_smooth,
                                double w_data,
                                double w_smooth)
{
    Eigen::VectorXd r_data, r_smooth;
    eval_residual_data(x, edgraph, key_old, key_new, key_indices, r_data, w_data);
    eval_residual_smooth(x, edgraph, r_smooth, w_smooth);

    r.resize(r_data.size() + r_smooth.size());
    r << r_data, r_smooth;

    cost_data   = 0.5 * r_data.squaredNorm();
    cost_smooth = 0.5 * r_smooth.squaredNorm();
}

void Optimizer::optimize(const Eigen::VectorXd& x0,
                         Eigen::VectorXd& x_opt,
                         EDGraph& edgraph,
                         const std::vector<Vec3>& key_old,
                         const std::vector<Vec3>& key_new,
                         const std::vector<int>& key_indices,
                         const Options& opt)
{
    const int DoF = static_cast<int>(x0.size());
    x_opt = x0;

    // Sanity: ensure neighbors exist if using smoothness
    if (edgraph.getNodeNeighbors().empty()) {
        if (opt.verbose) std::cout << "[Optimizer] Node neighbors empty — call buildKnnNeighbors() to enable smoothness.\n";
    }

    auto cost_total = [](const Eigen::VectorXd& r){ return 0.5 * r.squaredNorm(); };

    // Initial residual/cost
    Eigen::VectorXd r;
    double c_data=0, c_smooth=0;
    eval_residuals_full(x_opt, edgraph, key_old, key_new, key_indices, r, c_data, c_smooth, opt.w_data, opt.w_smooth);
    double cost = cost_total(r);
    if (opt.verbose) std::cout << "[LM] init: cost=" << cost << "  (data=" << c_data << ", smooth=" << c_smooth << ")\n";

    double lambda = opt.lambda_init;

    for (int it = 0; it < opt.max_iters; ++it) {
        // Central-difference Jacobian
        Eigen::MatrixXd J(r.size(), DoF);
        Eigen::VectorXd r_plus, r_minus;
        for (int j = 0; j < DoF; ++j) {
            Eigen::VectorXd x_p = x_opt; x_p(j) += opt.eps_jac;
            Eigen::VectorXd x_m = x_opt; x_m(j) -= opt.eps_jac;
            double d1, d2; // throwaways
            eval_residuals_full(x_p, edgraph, key_old, key_new, key_indices, r_plus, d1, d2, opt.w_data, opt.w_smooth);
            eval_residuals_full(x_m, edgraph, key_old, key_new, key_indices, r_minus, d1, d2, opt.w_data, opt.w_smooth);
            J.col(j) = (r_plus - r_minus) / (2.0 * opt.eps_jac);
        }

        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd g = -J.transpose() * r;
        H += lambda * Eigen::MatrixXd::Identity(DoF, DoF);

        Eigen::VectorXd dx = H.ldlt().solve(g);
        if (!dx.allFinite()) { std::cerr << "[LM] non-finite dx; abort.\n"; break; }
        if (dx.norm() < opt.tol_dx) {
            if (opt.verbose) std::cout << "[LM] Converged by |dx| at iter " << it << "\n";
            break;
        }

        // Try step
        Eigen::VectorXd x_new = x_opt + dx;
        Eigen::VectorXd r_new; double cd_new=0, cs_new=0;
        eval_residuals_full(x_new, edgraph, key_old, key_new, key_indices, r_new, cd_new, cs_new, opt.w_data, opt.w_smooth);
        double cost_new = cost_total(r_new);

        if (cost_new < cost) {
            // Accept
            x_opt = x_new; r = r_new; cost = cost_new; c_data = cd_new; c_smooth = cs_new;
            lambda = std::max(1e-10, lambda * 0.5);
            if (opt.verbose) std::cout << "[LM] it=" << it << "  cost=" << cost
                                       << "  (data=" << c_data << ", smooth=" << c_smooth << ")  lambda=" << lambda
                                       << "  (accept)\n";
        } else {
            // Reject and increase lambda; retry a few times
            bool accepted = false;
            for (int rep = 0; rep < 4; ++rep) {
                lambda *= 4.0;
                Eigen::MatrixXd H2 = J.transpose()*J + lambda * Eigen::MatrixXd::Identity(DoF, DoF);
                dx = H2.ldlt().solve(g);
                if (!dx.allFinite()) break;
                x_new = x_opt + dx;
                eval_residuals_full(x_new, edgraph, key_old, key_new, key_indices, r_new, cd_new, cs_new, opt.w_data, opt.w_smooth);
                cost_new = cost_total(r_new);
                if (cost_new < cost) { x_opt = x_new; r = r_new; cost = cost_new; c_data = cd_new; c_smooth = cs_new; accepted = true; break; }
            }
            if (opt.verbose) std::cout << "[LM] it=" << it << "  try_cost=" << cost_new << "  lambda=" << lambda
                                       << (accepted ? "  (accepted after damping)\n" : "  (rejected)\n");
            if (!accepted) break;
        }
    }

    // Update EDGraph state for downstream use
    edgraph.updateFromStateVector(x_opt, 0);
}