#ifndef EDGRAPH_H
#define EDGRAPH_H

#include <vector>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "MeshModel.h"

struct DeformationNode {
    Eigen::Vector3d position;     // node center (rest)
    Sophus::SE3d    transform;    // current SE3
};

class EDGraph {
public:
    explicit EDGraph(int K);

    // Build nodes by sampling every N-th vertex, then bind all vertices
    void initializeGraph(const std::vector<MeshModel::Vertex>& vertices,
                         int sampling_step);

    void setGraphNodes(const std::vector<DeformationNode>& nodes);

    // Bind each vertex to K nearest nodes (weights use (K+1)-th distance as base; then keep first K)
    void bindVertices(const std::vector<MeshModel::Vertex>& vertices);

    // Deform a single vertex (by its precomputed binding of vidx)
    Eigen::Vector3d deformVertex(const MeshModel::Vertex& vertex, int vidx) const;
    // Overload for convenience when you have Eigen::Vector3d instead of MeshModel::Vertex
    Eigen::Vector3d deformVertex(const Eigen::Vector3d& v, int vidx) const;

    // Evaluate deformation under an EXTERNAL state vector x (does NOT modify internal graph state)
    // ids/ws are the precomputed binding for the vertex to be deformed.
    Eigen::Vector3d deformVertexByState(const Eigen::Vector3d& v,
                                        const Eigen::VectorXd& x,
                                        const std::vector<int>& ids,
                                        const std::vector<double>& ws,
                                        int offset) const;

    // State I/O (6 DoF per node); write will auto-resize target vector
    void updateFromStateVector(const Eigen::VectorXd& x, int offset);
    void writeToStateVector(Eigen::VectorXd& x, int offset) const;

    int numNodes() const { return static_cast<int>(graph_.size()); }

    // Optional accessors (useful for debugging/comparison)
    const std::vector<DeformationNode>& getGraphNodes() const { return graph_; }
    const std::vector<std::vector<int>>&    getBindings() const { return bindings_; }
    const std::vector<std::vector<double>>& getWeights()  const { return weights_; }

private:
    int K_ = 4;  // number of bound nodes per vertex

    std::vector<DeformationNode>       graph_;     // nodes
    std::vector<std::vector<int>>      bindings_;  // per-vertex node ids
    std::vector<std::vector<double>>   weights_;   // per-vertex weights (sum to 1)
};

#endif // EDGRAPH_H