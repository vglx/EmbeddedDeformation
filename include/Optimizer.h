#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <Eigen/Dense>
#include "EDGraph.h"

class Optimizer {
public:
    struct Options {
        int    max_iters;
        double lambda_init;
        double eps_jac;
        double tol_dx;
        double w_data;
        double w_smooth;
        bool   verbose;
        // 用构造函数给默认值（跨编译器最稳）
        Options()
        : max_iters(50), lambda_init(1e-4), eps_jac(1e-7), tol_dx(1e-6),
          w_data(10.0), w_smooth(1.0), verbose(true) {}
    };

    Optimizer() = default;

    void optimize(const Eigen::VectorXd& x0,
                  Eigen::VectorXd& x_opt,
                  EDGraph& edgraph,
                  const std::vector<Eigen::Vector3d>& key_old,
                  const std::vector<Eigen::Vector3d>& key_new,
                  const std::vector<int>& key_indices,
                  const Options& opt);   // 注意：去掉了默认参数
};

#endif // OPTIMIZER_H