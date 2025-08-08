// Optimizer.cpp
#include "Optimizer.h"
#include <iostream>

void Optimizer::optimize(const Eigen::VectorXd& x0,
                         Eigen::VectorXd& x_opt,
                         EDGraph& edgraph,
                         const std::vector<Eigen::Vector3d>& key_old,
                         const std::vector<Eigen::Vector3d>& key_new) {
    const int max_iter = 10;
    const double lambda = 1e-3;
    const double eps = 1e-6;

    Eigen::VectorXd x = x0;
    const int num_keys = static_cast<int>(key_old.size());
    const int num_vars = x0.size();
    Eigen::VectorXd residuals(3 * num_keys);
    Eigen::MatrixXd J(3 * num_keys, num_vars);

    for (int iter = 0; iter < max_iter; ++iter) {
        edgraph.updateFromStateVector(x, 0);

        for (int i = 0; i < num_keys; ++i) {
            MeshModel::Vertex v;
            v.x = key_old[i].x();
            v.y = key_old[i].y();
            v.z = key_old[i].z();
            Eigen::Vector3d mapped = edgraph.deformVertex(v, -1);  // -1 表示自动最近邻搜索
            residuals.segment<3>(3 * i) = mapped - key_new[i];
        }

        for (int j = 0; j < num_vars; ++j) {
            Eigen::VectorXd x_pert = x;
            x_pert(j) += eps;
            edgraph.updateFromStateVector(x_pert, 0);

            for (int i = 0; i < num_keys; ++i) {
                MeshModel::Vertex v;
                v.x = key_old[i].x();
                v.y = key_old[i].y();
                v.z = key_old[i].z();
                Eigen::Vector3d mapped = edgraph.deformVertex(v, -1);
                Eigen::Vector3d r_plus = mapped - key_new[i];
                J.block<3, 1>(3 * i, j) = (r_plus - residuals.segment<3>(3 * i)) / eps;
            }
        }

        Eigen::MatrixXd H = J.transpose() * J + lambda * Eigen::MatrixXd::Identity(num_vars, num_vars);
        Eigen::VectorXd b = -J.transpose() * residuals;
        Eigen::VectorXd dx = H.ldlt().solve(b);

        x += dx;
        if (dx.norm() < 1e-5) break;
    }

    x_opt = x;
}