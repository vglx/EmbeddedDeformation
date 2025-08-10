#include "Optimizer.h"
#include <iostream>

// Helper: build residual vector r(x) for keypoint data term only
static void eval_residual_data(const Eigen::VectorXd& x,
                               const EDGraph& edgraph,
                               const std::vector<Eigen::Vector3d>& key_old,
                               const std::vector<Eigen::Vector3d>& key_new,
                               const std::vector<int>& key_indices,
                               Eigen::VectorXd& r,
                               double w_data = 10.0) // sqrt(100)
{
    const auto& B = edgraph.getBindings();
    const auto& W = edgraph.getWeights();
    const int Nk = static_cast<int>(key_old.size());
    r.resize(3 * Nk);

    for (int i = 0; i < Nk; ++i) {
        const int vid = key_indices[i];
        const auto& ids = B[vid];
        std::vector<double> ws = W[vid];
        // normalize defensively
        double s = 0.0; for (double v : ws) s += v; if (s > 0) for (double& v : ws) v /= s;

        Eigen::Vector3d pred = edgraph.deformVertexByState(key_old[i], x, ids, ws, /*offset=*/0);
        r.segment<3>(3*i) = w_data * (pred - key_new[i]);
    }
}

void Optimizer::optimize(const Eigen::VectorXd& x0,
                         Eigen::VectorXd& x_opt,
                         EDGraph& edgraph,
                         const std::vector<Eigen::Vector3d>& key_old,
                         const std::vector<Eigen::Vector3d>& key_new,
                         const std::vector<int>& key_indices)
{
    const int DoF = static_cast<int>(x0.size());
    x_opt = x0;

    const int max_iter = 30;
    double lambda = 1e-4;        // LM damping
    const double eps = 1e-6;     // FD step
    const double tol_dx = 1e-6;  // stop threshold

    auto cost_from_r = [](const Eigen::VectorXd& r){ return 0.5 * r.squaredNorm(); };

    // Initial residual & cost
    Eigen::VectorXd r;
    eval_residual_data(x_opt, edgraph, key_old, key_new, key_indices, r);
    double cost = cost_from_r(r);

    for (int it = 0; it < max_iter; ++it) {
        // Numeric Jacobian (data term only)
        Eigen::MatrixXd J(r.size(), DoF);
        for (int j = 0; j < DoF; ++j) {
            Eigen::VectorXd x_pert = x_opt;
            x_pert(j) += eps;
            Eigen::VectorXd r1;
            eval_residual_data(x_pert, edgraph, key_old, key_new, key_indices, r1);
            J.col(j) = (r1 - r) / eps;
        }

        // Normal equations with LM damping
        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd g = -J.transpose() * r;
        H += lambda * Eigen::MatrixXd::Identity(DoF, DoF);

        // Solve for step
        Eigen::VectorXd dx = H.ldlt().solve(g);
        if (!dx.allFinite()) {
            std::cerr << "[Optimizer] non-finite dx, abort.\n";
            break;
        }
        if (dx.norm() < tol_dx) {
            std::cout << "[Optimizer] Converged by |dx| at iter " << it << "\n";
            break;
        }

        // Try step
        Eigen::VectorXd x_new = x_opt + dx;
        Eigen::VectorXd r_new; eval_residual_data(x_new, edgraph, key_old, key_new, key_indices, r_new);
        double cost_new = cost_from_r(r_new);

        if (cost_new < cost) {
            // Accept, decrease lambda
            x_opt = x_new; r = r_new; cost = cost_new; lambda = std::max(1e-8, lambda * 0.5);
            std::cout << "[Optimizer] it=" << it << ", cost=" << cost << ", lambda=" << lambda << " (accept)\n";
        } else {
            // Reject, increase lambda and retry a few times
            bool accepted = false;
            for (int rep = 0; rep < 4; ++rep) {
                lambda *= 4.0;
                Eigen::MatrixXd H2 = J.transpose()*J + lambda * Eigen::MatrixXd::Identity(DoF, DoF);
                dx = H2.ldlt().solve(g);
                if (!dx.allFinite()) break;
                x_new = x_opt + dx;
                eval_residual_data(x_new, edgraph, key_old, key_new, key_indices, r_new);
                cost_new = cost_from_r(r_new);
                if (cost_new < cost) { x_opt = x_new; r = r_new; cost = cost_new; accepted = true; break; }
            }
            std::cout << "[Optimizer] it=" << it << ", cost_try=" << cost_new << ", lambda=" << lambda
                      << (accepted ? " (accepted after damping)\n" : " (rejected)\n");
            if (!accepted) break;
        }
    }

    // Update EDGraph state for downstream use
    edgraph.updateFromStateVector(x_opt, 0);
}