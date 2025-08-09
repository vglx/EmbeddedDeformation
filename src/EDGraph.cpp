#include "EDGraph.h"

#include <unordered_map>
#include <algorithm>
#include <cmath>

// 体素键（用于下采样）
struct VoxelKey {
    int x, y, z;
};
struct VoxelHash {
    size_t operator()(const VoxelKey& k) const noexcept {
        // 朴素哈希
        return (static_cast<size_t>(1469598103934665603ull) ^ (k.x*73856093) ^ (k.y*19349663) ^ (k.z*83492791));
    }
};
struct VoxelEq {
    bool operator()(const VoxelKey& a, const VoxelKey& b) const noexcept {
        return a.x==b.x && a.y==b.y && a.z==b.z;
    }
};

EDGraph::EDGraph(int K) : K_(K) {}

void EDGraph::setGraphNodes(const std::vector<DeformationNode>& nodes) {
    graph_ = nodes;
}

void EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& vertices,
                              double grid_size)
{
    // 1) 体素平均，得到节点位置
    std::unordered_map<VoxelKey, std::pair<Eigen::Vector3d,int>, VoxelHash, VoxelEq> voxel_map;
    voxel_map.reserve(vertices.size()/10 + 1);

    for (const auto& v : vertices) {
        VoxelKey key{
            static_cast<int>(std::floor(v.x / grid_size)),
            static_cast<int>(std::floor(v.y / grid_size)),
            static_cast<int>(std::floor(v.z / grid_size))
        };
        auto& slot = voxel_map[key];
        if (slot.second == 0) slot.first.setZero();
        slot.first += Eigen::Vector3d(v.x, v.y, v.z);
        slot.second += 1;
    }

    std::vector<DeformationNode> nodes;
    nodes.reserve(voxel_map.size());
    for (const auto& kv : voxel_map) {
        const Eigen::Vector3d c = kv.second.first / std::max(1, kv.second.second);
        DeformationNode dn;
        dn.position  = c;
        dn.transform = Sophus::SE3d(); // 单位变换
        nodes.push_back(dn);
    }

    setGraphNodes(nodes);

    // 2) 绑定顶点
    bindVertices(vertices);
}

void EDGraph::bindVertices(const std::vector<MeshModel::Vertex>& vertices) {
    const size_t nV = vertices.size();
    const int    G  = numNodes();
    bindings_.assign(nV, {});
    weights_.assign(nV, {});

    if (G == 0) return;

    for (size_t i = 0; i < nV; ++i) {
        Eigen::Vector3d v(vertices[i].x, vertices[i].y, vertices[i].z);
        std::vector<std::pair<int,double>> dists; // (node_id, dist2)
        dists.reserve(G);
        for (int j = 0; j < G; ++j) {
            double d2 = (graph_[j].position - v).squaredNorm();
            dists.emplace_back(j, std::sqrt(std::max(0.0, d2)));
        }
        std::nth_element(dists.begin(), dists.begin() + std::min(K_, (int)dists.size()), dists.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; });

        const int kth = std::min(K_, (int)dists.size());
        bindings_[i].resize(kth);
        weights_[i].resize(kth);

        // 以第 (kth-1) 个距离为基准；加入零保护
        double base = dists[std::max(0, kth - 1)].second;
        if (base < 1e-12) base = 1e-12;

        double sumw = 0.0;
        for (int k = 0; k < kth; ++k) {
            bindings_[i][k] = dists[k].first;
            const double raw = 1.0 - dists[k].second / base;  // 0..1 区间
            const double w   = std::max(0.0, raw);            // 负权裁 0
            weights_[i][k] = w;
            sumw += w;
        }
        if (sumw < 1e-12) {
            for (int k = 0; k < kth; ++k) weights_[i][k] = 1.0 / kth;
        } else {
            for (int k = 0; k < kth; ++k) weights_[i][k] /= sumw;
        }
    }
}

void EDGraph::buildKnnNeighbors(int Ksmooth) {
    const int G = numNodes();
    if (G == 0 || Ksmooth <= 0) return;
    for (int i = 0; i < G; ++i) {
        std::vector<std::pair<double,int>> d;
        d.reserve(G-1);
        for (int j = 0; j < G; ++j) if (j != i) {
            d.emplace_back((graph_[j].position - graph_[i].position).squaredNorm(), j);
        }
        if ((int)d.size() > Ksmooth) {
            std::nth_element(d.begin(), d.begin()+Ksmooth, d.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; });
            d.resize(Ksmooth);
        }
        graph_[i].neighbors.clear();
        graph_[i].neighbors.reserve(d.size());
        for (auto& pr : d) graph_[i].neighbors.push_back(pr.second);
    }
}

Eigen::Vector3d EDGraph::deformVertex(const Eigen::Vector3d& v, int vertex_index) const {
    const auto& ids = bindings_[vertex_index];
    const auto& ws  = weights_[vertex_index];
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    for (size_t k = 0; k < ids.size(); ++k) {
        const auto& node = graph_[ids[k]];
        const Eigen::Vector3d& g = node.position;
        out += ws[k] * (node.transform.so3() * (v - g) + g + node.transform.translation());
    }
    return out;
}

Eigen::Vector3d EDGraph::deformVertexByState(const Eigen::Vector3d& v,
                                             const Eigen::VectorXd& x,
                                             const std::vector<int>& node_ids,
                                             const std::vector<double>& node_ws,
                                             int offset) const
{
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    for (size_t k = 0; k < node_ids.size(); ++k) {
        const int j = node_ids[k];
        Eigen::Matrix<double,6,1> xi = x.segment<6>(offset + 6*j);
        Sophus::SE3d T = Sophus::SE3d::exp(xi);
        const Eigen::Vector3d& g = graph_[j].position;
        out += node_ws[k] * (T.so3() * (v - g) + g + T.translation());
    }
    return out;
}

void EDGraph::updateFromStateVector(const Eigen::VectorXd& x, int offset) {
    const int G = numNodes();
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix<double,6,1> se3 = x.segment<6>(offset + 6*i);
        graph_[i].transform = Sophus::SE3d::exp(se3);
    }
}

void EDGraph::writeToStateVector(Eigen::VectorXd& x, int offset) const {
    const int G = numNodes();
    if (x.size() < offset + 6*G) x.conservativeResize(offset + 6*G); // 自动扩容
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix<double,6,1> se3 = graph_[i].transform.log();
        x.segment<6>(offset + 6*i) = se3;
    }
}