#pragma once
#include <Eigen/Dense>
#include <vector>
#include "EDGraph.h"

class Optimizer {
public:
    struct Options {
        int    max_iters;      // LM iterations
        double lambda_init;    // initial damping
        double eps_jac;        // central-diff step for numeric J
        double tol_dx;         // step norm stop
        // residual weights (already on residual scale; i.e., sqrt of cost weights)
        double w_data;         // keypoint data term
        double w_smooth;       // node smoothness term
        double w_ortho;        // orthogonality regularization term
        bool   verbose;
        // Explicit default constructor to avoid NSDMI + default-arg issues on some compilers
        Options()
        : max_iters(60), lambda_init(1e-4), eps_jac(1e-7), tol_dx(1e-6),
          w_data(0.1), w_smooth(0.316), w_ortho(0.5), verbose(true) {}
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