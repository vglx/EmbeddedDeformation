#include "Optimizer.h"
#include <iostream>
#include <Eigen/Sparse>

void Optimizer::optimize(const Eigen::VectorXd& x0,
                         Eigen::VectorXd& x_opt,
                         EDGraph& edgraph,
                         const std::vector<Eigen::Vector3d>& key_old,
                         const std::vector<Eigen::Vector3d>& key_new,
                         const std::vector<int>& key_indices) {
    const double epsilon = 1e-6;
    const int max_iter = 20;
    double lambda = 1e-4;

    x_opt = x0;
    int num_vars = static_cast<int>(x0.size());
    int num_residuals = static_cast<int>(key_old.size()) * 3;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Build residual vector and Jacobian
        Eigen::VectorXd residual(num_residuals);
        Eigen::SparseMatrix<double> J(num_residuals, num_vars);
        std::vector<Eigen::Triplet<double>> triplets;

        for (size_t i = 0; i < key_old.size(); ++i) {
            // Compute deformed keypoint
            Eigen::Vector3d def = edgraph.deformVertexByState(key_old[i], x_opt, key_indices[i]);
            // Residual: difference to target
            Eigen::Vector3d r = def - key_new[i];
            residual.segment<3>(3 * i) = r;
            // Numerical Jacobian
            for (int j = 0; j < num_vars; ++j) {
                Eigen::VectorXd x_eps = x_opt;
                double delta = std::max(1e-8, std::abs(x_opt[j]) * 1e-4);
                x_eps[j] += delta;
                Eigen::Vector3d def_eps = edgraph.deformVertexByState(key_old[i], x_eps, key_indices[i]);
                Eigen::Vector3d diff = (def_eps - def) / delta;
                triplets.emplace_back(3 * i + 0, j, diff[0]);
                triplets.emplace_back(3 * i + 1, j, diff[1]);
                triplets.emplace_back(3 * i + 2, j, diff[2]);
            }
        }
        J.setFromTriplets(triplets.begin(), triplets.end());

        // Form approximated Hessian H = J^T J + λ I
        Eigen::SparseMatrix<double> H = J.transpose() * J;
        for (int k = 0; k < H.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(H, k); it; ++it)
                if (it.row() == it.col())
                    it.valueRef() += lambda;

        // Gradient g = -J^T r
        Eigen::VectorXd g = -J.transpose() * residual;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(H);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed" << std::endl;
            break;
        }
        Eigen::VectorXd dx = solver.solve(g);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solving failed" << std::endl;
            break;
        }

        // Check convergence
        if (dx.norm() < epsilon) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }

        // Update state
        x_opt += dx;
    }
}