#pragma once
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cmath>
#include "MeshModel.h"

// ---- Deformation node (Affine) ----
// Each node carries a center position (g), an affine 3x3 A, and a translation t
// v' = A * (v - g) + g + t
struct DeformationNode {
    Eigen::Vector3d position;   // g
    Eigen::Matrix3d A;          // 3x3 affine (rotation/scale/shear)
    Eigen::Vector3d t;          // translation

    DeformationNode()
        : position(Eigen::Vector3d::Zero()), A(Eigen::Matrix3d::Identity()), t(Eigen::Vector3d::Zero()) {}
};

class EDGraph {
public:
    explicit EDGraph(int K);

    // Build graph nodes via voxel downsampling, then bind vertices (K-NN with base weighting)
    void initializeGraph(const std::vector<MeshModel::Vertex>& vertices, double grid_size);

    // Directly set nodes (e.g., from MATLAB nodes.txt), then you should call bindVertices()
    void setGraphNodes(const std::vector<DeformationNode>& nodes);

    // Bind each mesh vertex to K nearest deformation nodes with base weighting and normalization
    void bindVertices(const std::vector<MeshModel::Vertex>& vertices);

    // Build K-NN neighbors over nodes for smoothness priors
    void buildKnnNeighbors(int Ksmooth);

    // Deform a mesh vertex (using stored node states A,t) by its precomputed weights
    Eigen::Vector3d deformVertex(const MeshModel::Vertex& vertex, int vidx) const;
    // Deform a 3D point (using stored node states A,t) by its precomputed weights
    Eigen::Vector3d deformVertex(const Eigen::Vector3d& v, int vidx) const;

    // Deform a 3D point using a *state vector x* instead of stored states (used by optimizer)
    // x packs [A(3x3)=9, t(3)=3] per node, row-major for A: [a11,a12,a13,a21,...,a33, tx,ty,tz]
    Eigen::Vector3d deformVertexByState(const Eigen::Vector3d& v,
                                        const Eigen::VectorXd& x,
                                        const std::vector<int>& ids,
                                        const std::vector<double>& ws,
                                        int offset) const;

    // Update internal nodes (A,t) from a state vector x (12 DoF per node)
    void updateFromStateVector(const Eigen::VectorXd& x, int offset);

    // Write internal nodes (A,t) into a state vector x (12 DoF per node)
    void writeToStateVector(Eigen::VectorXd& x, int offset) const;

    // ---- Getters used by optimizer / main ----
    int numNodes() const { return static_cast<int>(graph_.size()); }

    const std::vector<std::vector<int>>& getBindings() const { return bindings_; }
    const std::vector<std::vector<double>>& getWeights() const { return weights_; }
    const std::vector<DeformationNode>& getGraphNodes() const { return graph_; }
    const std::vector<std::vector<int>>& getNodeNeighbors() const { return node_neighbors_; }

private:
    // Helper: voxel downsample to create nodes (positions = voxel means)
    void voxelDownsample(const std::vector<MeshModel::Vertex>& vertices,
                         double grid_size,
                         std::vector<DeformationNode>& out_nodes) const;

private:
    int K_;  // number of bindings per vertex (K-nearest)

    // Graph state
    std::vector<DeformationNode> graph_;              // nodes (A,t stored here)
    std::vector<std::vector<int>> node_neighbors_;    // K-NN over nodes for smoothness

    // Bindings
    std::vector<std::vector<int>> bindings_;          // per-vertex bound node indices
    std::vector<std::vector<double>> weights_;        // per-vertex normalized weights
};