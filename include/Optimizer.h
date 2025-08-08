#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <Eigen/Dense>
#include "EDGraph.h"

class Optimizer {
public:
    void optimize(const Eigen::VectorXd& x0,
                  Eigen::VectorXd& x_opt,
                  EDGraph& edgraph,
                  const std::vector<Eigen::Vector3d>& key_old,
                  const std::vector<Eigen::Vector3d>& key_new,
                  const std::vector<int>& key_indices);
};

#endif // OPTIMIZER_H