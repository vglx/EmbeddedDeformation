#ifndef EDGRAPH_H
#define EDGRAPH_H

#include <vector>
#include <unordered_map>
#include <tuple>
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

    // Build nodes by voxel downsampling (grid_size in same unit as vertices), then bind all vertices
    void initializeGraph(const std::vector<MeshModel::Vertex>& vertices,
                         double grid_size);

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

    // Build KNN neighbors per node for smoothness term; typical Ksmooth = num_nearestpts-1 (e.g., 5 when K=6)
    void buildKnnNeighbors(int Ksmooth);

    // State I/O (6 DoF per node); write will auto-resize target vector
    void updateFromStateVector(const Eigen::VectorXd& x, int offset);
    void writeToStateVector(Eigen::VectorXd& x, int offset) const;

    int numNodes() const { return static_cast<int>(graph_.size()); }

    // Optional accessors (useful for debugging/comparison)
    const std::vector<DeformationNode>& getGraphNodes() const { return graph_; }
    const std::vector<std::vector<int>>&    getBindings() const { return bindings_; }
    const std::vector<std::vector<double>>& getWeights()  const { return weights_; }
    const std::vector<std::vector<int>>&    getNodeNeighbors() const { return node_neighbors_; }

private:
    // --- voxel downsample internals ---
    struct VoxelKey { int x, y, z; };
    struct VoxelKeyHash {
        std::size_t operator()(const VoxelKey& k) const noexcept {
            // 3D hash (x,y,z) -> size_t
            // Use 64-bit mix; assume int fits voxel indices
            const uint64_t p1 = 73856093u, p2 = 19349663u, p3 = 83492791u;
            // cast to uint
            uint64_t ux = static_cast<uint64_t>(static_cast<uint32_t>(k.x));
            uint64_t uy = static_cast<uint64_t>(static_cast<uint32_t>(k.y));
            uint64_t uz = static_cast<uint64_t>(static_cast<uint32_t>(k.z));
            return static_cast<std::size_t>((ux * p1) ^ (uy * p2) ^ (uz * p3));
        }
    };
    struct VoxelKeyEq {
        bool operator()(const VoxelKey& a, const VoxelKey& b) const noexcept {
            return a.x==b.x && a.y==b.y && a.z==b.z;
        }
    };

    void voxelDownsample(const std::vector<MeshModel::Vertex>& vertices,
                         double grid_size,
                         std::vector<DeformationNode>& out_nodes) const;

private:
    int K_ = 4;  // number of bound nodes per vertex

    std::vector<DeformationNode>       graph_;          // nodes
    std::vector<std::vector<int>>      bindings_;       // per-vertex node ids
    std::vector<std::vector<double>>   weights_;        // per-vertex weights (sum to 1)
    std::vector<std::vector<int>>      node_neighbors_; // per-node neighbor ids for smoothness
};

#endif // EDGRAPH_H