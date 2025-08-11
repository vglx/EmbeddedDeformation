#include "Optimizer.h"
#include <iostream>
#include <numeric>

using Vec3 = Eigen::Vector3d;

namespace {
// r_data = w_data * (ED_x(key_old) - key_new)
void eval_residual_data(const Eigen::VectorXd& x,
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
        double s = 0.0; for (double v : ws) s += v; if (s>0) for (double& v : ws) v /= s; // defensive renorm
        Vec3 pred = edgraph.deformVertexByState(key_old[i], x, ids, ws, /*offset=*/0);
        r_data.segment<3>(3*i) = w_data * (pred - key_new[i]);
    }
}

// Smoothness over node graph edges (i<j):
// r_ij = A_i*(g_j - g_i) + g_i + t_i - (g_j + t_j)
void eval_residual_smooth(const Eigen::VectorXd& x,
                          const EDGraph& edgraph,
                          Eigen::VectorXd& r_smooth,
                          double w_smooth)
{
    const int G = edgraph.numNodes();
    const auto& nodes = edgraph.getGraphNodes();
    const auto& neigh = edgraph.getNodeNeighbors();

    // count undirected edges
    int E = 0; for (int i = 0; i < G; ++i) for (int j : neigh[i]) if (j > i) ++E;
    r_smooth.resize(3 * E);

    auto read_node = [&](int idx, Eigen::Matrix3d& A, Vec3& t){
        const int base = 12 * idx;
        A << x(base+0), x(base+1), x(base+2),
             x(base+3), x(base+4), x(base+5),
             x(base+6), x(base+7), x(base+8);
        t = Vec3(x(base+9), x(base+10), x(base+11));
    };

    int e = 0;
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix3d Ai; Vec3 ti; read_node(i, Ai, ti);
        const Vec3 gi = nodes[i].position;
        for (int j : neigh[i]) if (j > i) {
            Eigen::Matrix3d Aj; Vec3 tj; read_node(j, Aj, tj);
            const Vec3 gj = nodes[j].position;
            Vec3 lhs = Ai * (gj - gi) + gi + ti;
            Vec3 rhs = gj + tj;
            r_smooth.segment<3>(3*e) = w_smooth * (lhs - rhs);
            ++e;
        }
    }
}

// Orthogonality regularization per node:
// For each node i, r_ortho_i = vec(A_i^T A_i - I)  (9 residuals)
void eval_residual_ortho(const Eigen::VectorXd& x,
                         const EDGraph& edgraph,
                         Eigen::VectorXd& r_ortho,
                         double w_ortho)
{
    const int G = edgraph.numNodes();
    r_ortho.resize(9 * G);

    for (int i = 0; i < G; ++i) {
        const int base = 12 * i;
        Eigen::Matrix3d A;
        A << x(base+0), x(base+1), x(base+2),
             x(base+3), x(base+4), x(base+5),
             x(base+6), x(base+7), x(base+8);
        Eigen::Matrix3d M = A.transpose() * A - Eigen::Matrix3d::Identity();
        // vec in row-major order
        r_ortho.segment<9>(9*i) << M(0,0), M(0,1), M(0,2),
                                   M(1,0), M(1,1), M(1,2),
                                   M(2,0), M(2,1), M(2,2);
    }
    r_ortho *= w_ortho;
}

// Build full residual r = [r_data; r_smooth; r_ortho]
void eval_residuals_full(const Eigen::VectorXd& x,
                         const EDGraph& edgraph,
                         const std::vector<Vec3>& key_old,
                         const std::vector<Vec3>& key_new,
                         const std::vector<int>& key_indices,
                         Eigen::VectorXd& r,
                         double& cost_data,
                         double& cost_smooth,
                         double& cost_ortho,
                         double w_data,
                         double w_smooth,
                         double w_ortho)
{
    Eigen::VectorXd r_data, r_smooth, r_ortho;
    eval_residual_data(x, edgraph, key_old, key_new, key_indices, r_data,  w_data);
    eval_residual_smooth(x, edgraph, r_smooth, w_smooth);
    eval_residual_ortho(x, edgraph, r_ortho, w_ortho);

    r.resize(r_data.size() + r_smooth.size() + r_ortho.size());
    r << r_data, r_smooth, r_ortho;

    cost_data   = 0.5 * r_data.squaredNorm();
    cost_smooth = 0.5 * r_smooth.squaredNorm();
    cost_ortho  = 0.5 * r_ortho.squaredNorm();
}
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

    auto cost_total = [](const Eigen::VectorXd& r){ return 0.5 * r.squaredNorm(); };

    // Initial residual/cost
    Eigen::VectorXd r;
    double c_data=0, c_smooth=0, c_ortho=0;
    eval_residuals_full(x_opt, edgraph, key_old, key_new, key_indices,
                        r, c_data, c_smooth, c_ortho,
                        opt.w_data, opt.w_smooth, opt.w_ortho);
    double cost = cost_total(r);
    if (opt.verbose) std::cout << "[LM] init: cost=" << cost
                               << "  (data=" << c_data
                               << ", smooth=" << c_smooth
                               << ", ortho=" << c_ortho << ")\n";

    double lambda = opt.lambda_init;

    for (int it = 0; it < opt.max_iters; ++it) {
        // Central-difference Jacobian
        Eigen::MatrixXd J(r.size(), DoF);
        Eigen::VectorXd r_plus, r_minus;
        for (int j = 0; j < DoF; ++j) {
            Eigen::VectorXd x_p = x_opt; x_p(j) += opt.eps_jac;
            Eigen::VectorXd x_m = x_opt; x_m(j) -= opt.eps_jac;
            double d1, d2, d3; // throwaways
            eval_residuals_full(x_p, edgraph, key_old, key_new, key_indices,
                                r_plus, d1, d2, d3,
                                opt.w_data, opt.w_smooth, opt.w_ortho);
            eval_residuals_full(x_m, edgraph, key_old, key_new, key_indices,
                                r_minus, d1, d2, d3,
                                opt.w_data, opt.w_smooth, opt.w_ortho);
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
        Eigen::VectorXd r_new; double cd_new=0, cs_new=0, co_new=0;
        eval_residuals_full(x_new, edgraph, key_old, key_new, key_indices,
                            r_new, cd_new, cs_new, co_new,
                            opt.w_data, opt.w_smooth, opt.w_ortho);
        double cost_new = cost_total(r_new);

        if (cost_new < cost) {
            // Accept
            x_opt = x_new; r = r_new; cost = cost_new;
            c_data = cd_new; c_smooth = cs_new; c_ortho = co_new;
            lambda = std::max(1e-10, lambda * 0.5);
            if (opt.verbose) std::cout << "[LM] it=" << it
                                       << "  cost=" << cost
                                       << "  (data=" << c_data
                                       << ", smooth=" << c_smooth
                                       << ", ortho=" << c_ortho << ")"
                                       << "  lambda=" << lambda
                                       << "  (accept)\n";
        } else {
            // Reject and increase lambda; retry a few times
            bool accepted = false; double try_cost = cost_new;
            for (int rep = 0; rep < 4; ++rep) {
                lambda *= 4.0;
                Eigen::MatrixXd H2 = J.transpose()*J + lambda * Eigen::MatrixXd::Identity(DoF, DoF);
                dx = H2.ldlt().solve(g);
                if (!dx.allFinite()) break;
                x_new = x_opt + dx;
                eval_residuals_full(x_new, edgraph, key_old, key_new, key_indices,
                                    r_new, cd_new, cs_new, co_new,
                                    opt.w_data, opt.w_smooth, opt.w_ortho);
                try_cost = cost_total(r_new);
                if (try_cost < cost) { x_opt = x_new; r = r_new; cost = try_cost;
                    c_data = cd_new; c_smooth = cs_new; c_ortho = co_new; accepted = true; break; }
            }
            if (opt.verbose) std::cout << "[LM] it=" << it
                                       << "  try_cost=" << try_cost
                                       << "  lambda=" << lambda
                                       << (accepted ? "  (accepted after damping)\n" : "  (rejected)\n");
            if (!accepted) break;
        }
    }

    // Update EDGraph state for downstream use
    edgraph.updateFromStateVector(x_opt, 0);
}