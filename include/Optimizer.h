#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <Eigen/Dense>
#include "EDGraph.h"

class Optimizer {
public:
    struct Options {
        int    max_iters   = 50;     // LM iterations
        double lambda_init = 1e-4;   // initial damping
        double eps_jac     = 1e-7;   // central difference step
        double tol_dx      = 1e-6;   // step tolerance
        double w_data      = 10.0;   // sqrt-weight for keypoint term
        double w_smooth    = 1.0;    // sqrt-weight for smoothness term
        bool   verbose     = true;
    };

    Optimizer() = default;

    void optimize(const Eigen::VectorXd& x0,
                  Eigen::VectorXd& x_opt,
                  EDGraph& edgraph,
                  const std::vector<Eigen::Vector3d>& key_old,
                  const std::vector<Eigen::Vector3d>& key_new,
                  const std::vector<int>& key_indices,
                  const Options& opt = Options());
};

#endif // OPTIMIZER_H