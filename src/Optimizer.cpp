#include "Optimizer.h"
#include <iostream>
#include <limits>

namespace {
inline double cost_from_r(const Eigen::VectorXd& r) {
    return 0.5 * r.squaredNorm();
}
}

void Optimizer::optimize(const Eigen::VectorXd& x0,
                         Eigen::VectorXd& x_opt,
                         EDGraph& edgraph,
                         const std::vector<Eigen::Vector3d>& key_old,
                         const std::vector<Eigen::Vector3d>& key_new,
                         const std::vector<int>& key_indices)
{
    // --- Settings ---
    const int    G       = edgraph.numNodes();
    const int    DoF     = 6 * G;
    const int    max_it  = 30;
    const double tol_dx  = 1e-6;
    const double tol_dF  = 1e-9;

    // Residual weights (align roughly with MATLAB script: data > smooth)
    const double w_data   = 10.0;   // sqrt(100)
    const double w_smooth = 3.1622776601683795; // sqrt(10)

    x_opt = x0;

    // Accessors
    const auto& Nodes    = edgraph.getGraphNodes();
    const auto& Bindings = edgraph.getBindings();
    const auto& Weights  = edgraph.getWeights();

    // Build residual evaluation (smooth + data)
    auto eval_residual = [&](const Eigen::VectorXd& x, Eigen::VectorXd& r) {
        // Count directed smooth edges
        size_t E = 0;
        for (int i = 0; i < G; ++i) E += Nodes[i].neighbors.size();
        const int m = static_cast<int>(3*E + 3*key_old.size());
        r.setZero(m);

        int cursor = 0;

        // Smoothness residuals: r_ij = Ri(gj-gi) + gi + ti - (gj + tj)
        for (int i = 0; i < G; ++i) {
            Eigen::Matrix<double,6,1> xi = x.segment<6>(6*i);
            Sophus::SE3d Ti = Sophus::SE3d::exp(xi);
            const Eigen::Vector3d& gi = Nodes[i].position;
            for (int j : Nodes[i].neighbors) {
                Eigen::Matrix<double,6,1> xj = x.segment<6>(6*j);
                Sophus::SE3d Tj = Sophus::SE3d::exp(xj);
                const Eigen::Vector3d& gj = Nodes[j].position;

                Eigen::Vector3d pred_i = Ti.so3() * (gj - gi) + gi + Ti.translation();
                Eigen::Vector3d pred_j = gj + Tj.translation();
                r.segment<3>(cursor) = w_smooth * (pred_i - pred_j);
                cursor += 3;
            }
        }

        // Data residuals: r_k = ED(v_k) - v'_k, using per-vertex bindings/weights
        for (size_t k = 0; k < key_old.size(); ++k) {
            const int vid = key_indices[k];
            const auto& ids = Bindings[vid];
            std::vector<double> ws = Weights[vid];
            double s = 0.0; for (double w : ws) s += w; if (s > 0) for (double& w : ws) w /= s;

            Eigen::Vector3d pred = edgraph.deformVertexByState(key_old[k], x, ids, ws, /*offset=*/0);
            r.segment<3>(cursor) = w_data * (pred - key_new[k]);
            cursor += 3;
        }
    };

    // Finite-difference Jacobian (robust but slower). You can swap to analytic later.
    auto eval_numeric_J = [&](const Eigen::VectorXd& x, const Eigen::VectorXd& r0, Eigen::MatrixXd& J) {
        const double eps = 1e-6;
        const int m = static_cast<int>(r0.size());
        J.setZero(m, DoF);
        Eigen::VectorXd r1(m);
        for (int c = 0; c < DoF; ++c) {
            Eigen::VectorXd x_pert = x;
            x_pert[c] += eps;
            eval_residual(x_pert, r1);
            J.col(c) = (r1 - r0) / eps;
        }
    };

    // Levenberg–Marquardt style damping
    double lambda = 1e-4;

    // Initial residual & cost
    Eigen::VectorXd r;
    eval_residual(x_opt, r);
    double cost = cost_from_r(r);

    for (int it = 0; it < max_it; ++it) {
        // Jacobian at current x
        Eigen::MatrixXd J;
        eval_numeric_J(x_opt, r, J);

        // Normal equations with damping
        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd g = -J.transpose() * r;
        H += lambda * Eigen::MatrixXd::Identity(DoF, DoF);

        // Solve H dx = g
        Eigen::VectorXd dx = H.ldlt().solve(g);
        if (!dx.allFinite()) {
            std::cerr << "[Optimizer] Non-finite dx, abort.\n";
            break;
        }

        if (dx.norm() < tol_dx) {
            std::cout << "[Optimizer] Converged by dx at iter " << it << "\n";
            break;
        }

        // Tentative update
        Eigen::VectorXd x_new = x_opt + dx;
        Eigen::VectorXd r_new;
        eval_residual(x_new, r_new);
        double cost_new = cost_from_r(r_new);

        // Accept / reject step with simple LM schedule
        if (cost_new < cost - tol_dF) {
            // Good step: accept and decrease lambda
            x_opt = x_new;
            r = r_new;
            cost = cost_new;
            lambda = std::max(1e-8, lambda * 0.5);
            std::cout << "[Optimizer] it=" << it << ", cost=" << cost << ", lambda=" << lambda << " (accepted)\n";
        } else {
            // Bad step: increase lambda and retry (up to a few times)
            bool accepted = false;
            for (int rep = 0; rep < 5; ++rep) {
                lambda *= 4.0;
                H = J.transpose() * J + lambda * Eigen::MatrixXd::Identity(DoF, DoF);
                dx = H.ldlt().solve(g);
                if (!dx.allFinite() || dx.norm() < 1e-15) break;
                x_new = x_opt + dx;
                eval_residual(x_new, r_new);
                cost_new = cost_from_r(r_new);
                if (cost_new < cost - tol_dF) {
                    x_opt = x_new; r = r_new; cost = cost_new; accepted = true; break;
                }
            }
            std::cout << "[Optimizer] it=" << it << ", cost_try=" << cost_new << ", lambda=" << lambda
                      << (accepted ? " (accepted after damping)\n" : " (rejected)\n");
            if (!accepted) {
                // If we consistently fail to improve, stop.
                break;
            }
        }
    }
}