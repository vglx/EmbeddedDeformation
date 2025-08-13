#pragma once
#include <Eigen/Core>
#include <vector>
#include <functional>
#include "EDGraph.h"

struct OptimizerOptions {
    int    max_iters   = 50;
    bool   verbose     = true;

    // Line search (match MATLAB script semantics)
    double alpha0      = 0.5;   // initial step
    double step0       = 0.25;  // step adjust
    double gamma1      = 0.1;   // Armijo-like
    double gamma2      = 0.9;   // curvature-like

    // Termination threshold on F'PF and on change of F'PF (match MATLAB intent)
    double tol_cost    = 1e-9;

    // P diagonal weights (exactly mirror MATLAB Gauss_Newton_Optimization)
    // v_diag: per node -> first 6 rows = 1.0; next 3*num_nearestpts rows = 0.1
    // data rows (3 per control point) = 0.01
    double w_rot_rows  = 1.0;   // rotation rows (6 per node)
    double w_conn_rows = 0.1;   // connection rows (3*num_nearestpts per node in MATLAB code)
    double w_data_rows = 0.01;  // data rows (3 per control point)
};

class Optimizer {
public:
    explicit Optimizer(const OptimizerOptions& opt);

    // x: 12*G vector, per node (row-major A 9 + t 3)
    void optimize(EDGraph& edgraph,
                  Eigen::VectorXd& x, // in/out
                  const std::vector<Eigen::Vector3d>& key_old,
                  const std::vector<Eigen::Vector3d>& key_new,
                  const std::vector<int>&             key_indices);

private:
    const OptimizerOptions opt_;

    // Build F in exact MATLAB order and P diagonal
    void buildResidualVector(const Eigen::VectorXd& x,
                             const EDGraph& ed,
                             const std::vector<Eigen::Vector3d>& key_old,
                             const std::vector<Eigen::Vector3d>& key_new,
                             const std::vector<int>& key_idx,
                             Eigen::VectorXd& F,
                             Eigen::VectorXd& Pdiag,
                             int& num_rownode,
                             int& num_nodes,
                             int& num_ctrl) const;

    // Numerical Jacobian of full F (central diff)
    void numericJacobian(std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> f,
                         const Eigen::VectorXd& x,
                         Eigen::MatrixXd& J,
                         double eps = 1e-6) const;
};