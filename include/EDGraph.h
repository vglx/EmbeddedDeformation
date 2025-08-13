#pragma once
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cmath>
#include "MeshModel.h"

// Deformation node
// v' = A * (v - g) + g + t
struct DeformationNode {
    Eigen::Vector3d position;   // g
    Eigen::Matrix3d A;          // 3x3 affine
    Eigen::Vector3d t;          // translation
    DeformationNode()
        : position(Eigen::Vector3d::Zero()), A(Eigen::Matrix3d::Identity()), t(Eigen::Vector3d::Zero()) {}
};

class EDGraph {
public:
    explicit EDGraph(int K);

    // Build graph nodes via voxel downsampling, then bind vertices (K-NN with MATLAB K+1 base weighting)
    void initializeGraph(const std::vector<MeshModel::Vertex>& vertices, double grid_size);

    // Directly set nodes (then call bindVertices)
    void setGraphNodes(const std::vector<DeformationNode>& nodes);

    // Bind each mesh vertex to K nearest deformation nodes with base weighting and row-normalization
    void bindVertices(const std::vector<MeshModel::Vertex>& vertices);

    // Build K-NN neighbors over nodes for smoothness (typically Ksmooth = K - 1)
    void buildKnnNeighbors(int Ksmooth);

    // Deform a vertex by stored states
    Eigen::Vector3d deformVertex(const MeshModel::Vertex& vertex, int vidx) const;
    Eigen::Vector3d deformVertex(const Eigen::Vector3d& v, int vidx) const;

    // Deform a 3D point using a state vector x (optimizer path)
    Eigen::Vector3d deformVertexByState(const Eigen::Vector3d& v,
                                        const Eigen::VectorXd& x,
                                        const std::vector<int>& ids,
                                        const std::vector<double>& ws,
                                        int offset) const;

    // Update/read internal nodes (A,t) from/to state vector x (12 DoF per node)
    void updateFromStateVector(const Eigen::VectorXd& x, int offset);
    void writeToStateVector(Eigen::VectorXd& x, int offset) const;

    // ---- Getters ----
    int numNodes() const { return static_cast<int>(graph_.size()); }

    // Original getter names (kept)
    const std::vector<std::vector<int>>&    getBindings() const { return bindings_; }
    const std::vector<std::vector<double>>& getWeights()  const { return weights_; }

    // Aliases to match optimizer expectations
    const std::vector<std::vector<int>>&    getVertexBindings() const { return bindings_; }
    const std::vector<std::vector<double>>& getVertexWeights()  const { return weights_; }

    const std::vector<DeformationNode>& getGraphNodes()   const { return graph_; }
    const std::vector<std::vector<int>>& getNodeNeighbors() const { return node_neighbors_; }

private:
    // Voxel downsample node positions (voxel means)
    void voxelDownsample(const std::vector<MeshModel::Vertex>& vertices,
                         double grid_size,
                         std::vector<DeformationNode>& out_nodes) const;

private:
    int K_;  // number of bindings per vertex

    // Graph state
    std::vector<DeformationNode>   graph_;
    std::vector<std::vector<int>>  node_neighbors_; // for smoothness

    // Vertex bindings
    std::vector<std::vector<int>>    bindings_; // per-vertex node ids (size K_ or fewer)
    std::vector<std::vector<double>> weights_;  // per-vertex weights (row-normalized)
};