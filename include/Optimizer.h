#pragma once
#include <Eigen/Dense>
#include <vector>
#include "EDGraph.h"

class Optimizer {
public:
    struct Options {
        int    max_iters    = 60;      // LM iterations
        double lambda_init  = 1e-4;    // initial damping
        double eps_jac      = 1e-7;    // central-diff step for numeric J
        double tol_dx       = 1e-6;    // step norm stop
        // residual weights (already on residual scale; i.e., sqrt of cost weights)
        double w_data       = 0.1;     // keypoint data term
        double w_smooth     = 0.316;   // node smoothness term
        double w_ortho      = 0.5;     // orthogonality regularization term
        bool   verbose      = true;
    };

    // Optimize node states x (12*G DoF: A(3x3 row-major) + t(3) per node) for affine ED
    // key_old/new: corresponding 3D pairs; key_indices: which mesh vertex each keypoint binds to
    void optimize(const Eigen::VectorXd& x0,
                  Eigen::VectorXd& x_opt,
                  EDGraph& edgraph,
                  const std::vector<Eigen::Vector3d>& key_old,
                  const std::vector<Eigen::Vector3d>& key_new,
                  const std::vector<int>& key_indices,
                  const Options& opt = Options());
};