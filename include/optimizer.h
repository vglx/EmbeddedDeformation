// Optimizer.h
#pragma once
#include <Eigen/Dense>
#include <vector>
#include "EDGraph.h"

class Optimizer {
public:
    void optimize(const Eigen::VectorXd& x0,
                  Eigen::VectorXd& x_opt,
                  EDGraph& edgraph,
                  const std::vector<Eigen::Vector3d>& key_old,
                  const std::vector<Eigen::Vector3d>& key_new);
};