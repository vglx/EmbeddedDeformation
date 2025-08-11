#include "EDGraph.h"
#include <algorithm>
#include <cmath>

EDGraph::EDGraph(int K): K_(K) {}

void EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& vertices, double grid_size) {
    // 1) Voxel downsample to build nodes (match MATLAB grid_size behavior)
    std::vector<DeformationNode> nodes;
    voxelDownsample(vertices, grid_size, nodes);
    setGraphNodes(nodes);

    // 2) Bind vertices to nodes (K+1 base weighting)
    bindVertices(vertices);
}

void EDGraph::setGraphNodes(const std::vector<DeformationNode>& nodes) {
    graph_ = nodes;
}

void EDGraph::voxelDownsample(const std::vector<MeshModel::Vertex>& vertices,
                              double grid_size,
                              std::vector<DeformationNode>& out_nodes) const
{
    out_nodes.clear();
    if (vertices.empty()) return;
    const double gs = (grid_size > 1e-12) ? grid_size : 1.0;

    struct Accum { Eigen::Vector3d sum; int cnt; };
    std::unordered_map<VoxelKey, Accum, VoxelKeyHash, VoxelKeyEq> buckets;
    buckets.reserve(vertices.size()/8 + 1);

    auto toKey = [gs](const Eigen::Vector3d& p){
        // floor division into voxel indices; handle negatives consistently
        int ix = static_cast<int>(std::floor(p.x() / gs));
        int iy = static_cast<int>(std::floor(p.y() / gs));
        int iz = static_cast<int>(std::floor(p.z() / gs));
        return VoxelKey{ix,iy,iz};
    };

    // accumulate
    for (const auto& v : vertices) {
        Eigen::Vector3d p(v.x, v.y, v.z);
        VoxelKey key = toKey(p);
        auto it = buckets.find(key);
        if (it == buckets.end()) {
            buckets.emplace(key, Accum{p, 1});
        } else {
            it->second.sum += p; it->second.cnt += 1;
        }
    }

    out_nodes.reserve(buckets.size());
    for (const auto& kv : buckets) {
        DeformationNode node;
        node.position  = kv.second.sum / std::max(1, kv.second.cnt);
        node.transform = Sophus::SE3d(); // identity
        out_nodes.push_back(node);
    }
}

void EDGraph::bindVertices(const std::vector<MeshModel::Vertex>& vertices) {
    const size_t nV = vertices.size();
    const int    G  = static_cast<int>(graph_.size());
    bindings_.assign(nV, {});
    weights_.assign(nV, {});
    if (G == 0) return;

    for (size_t i = 0; i < nV; ++i) {
        const Eigen::Vector3d v(vertices[i].x, vertices[i].y, vertices[i].z);

        // 收集到所有节点的距离
        std::vector<std::pair<int,double>> dists;
        dists.reserve(G);
        for (int j = 0; j < G; ++j) {
            double dist = (v - graph_[j].position).norm();
            dists.emplace_back(j, dist);
        }

        // 计算 base：第 K+1 小的距离（0-based 第 K_ 位）
        const int base_rank = std::min(K_, G - 1);  // 防越界
        std::nth_element(
            dists.begin(), dists.begin() + base_rank, dists.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; }
        );
        double base = dists[base_rank].second;
        if (base < 1e-12) base = 1e-12; // 防除零

        // 确保前 K_ 个是真正最近的 K 个（提升稳定性）
        const int kth = std::min(K_, static_cast<int>(dists.size()));
        std::partial_sort(
            dists.begin(), dists.begin() + kth, dists.end(),
            [](const auto& a, const auto& b){ return a.second < b.second; }
        );

        // 计算权重：w_k = max(0, 1 - d_k / base)，然后归一化
        bindings_[i].resize(kth);
        weights_[i].resize(kth);
        double sumw = 0.0;
        for (int k = 0; k < kth; ++k) {
            bindings_[i][k] = dists[k].first;
            const double raw = 1.0 - dists[k].second / base;
            const double w   = std::max(0.0, raw);
            weights_[i][k]   = w;
            sumw += w;
        }
        if (sumw < 1e-12) {
            // 退化情况：均匀分布
            const double uni = 1.0 / std::max(1, kth);
            for (double& w : weights_[i]) w = uni;
        } else {
            for (double& w : weights_[i]) w /= sumw;
        }
    }
}

Eigen::Vector3d EDGraph::deformVertex(const MeshModel::Vertex& vertex, int vidx) const {
    const Eigen::Vector3d v(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d out = Eigen::Vector3d::Zero();

    const auto& ids = bindings_[vidx];
    const auto& ws  = weights_[vidx];
    double sw = 0.0; for (double w : ws) sw += w; const double inv_sw = (sw>1e-12)?1.0/sw:1.0;
    for (size_t k = 0; k < ids.size(); ++k) {
        const auto& node = graph_[ids[k]];
        const Eigen::Vector3d& g = node.position;
        out += (ws[k]*inv_sw) * (node.transform.so3() * (v - g) + g + node.transform.translation());
    }
    return out;
}

Eigen::Vector3d EDGraph::deformVertex(const Eigen::Vector3d& v, int vidx) const {
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    const auto& ids = bindings_[vidx];
    const auto& ws  = weights_[vidx];
    double sw = 0.0; for (double w : ws) sw += w; const double inv_sw = (sw>1e-12)?1.0/sw:1.0;
    for (size_t k = 0; k < ids.size(); ++k) {
        const auto& node = graph_[ids[k]];
        const Eigen::Vector3d& g = node.position;
        out += (ws[k]*inv_sw) * (node.transform.so3() * (v - g) + g + node.transform.translation());
    }
    return out;
}

Eigen::Vector3d EDGraph::deformVertexByState(const Eigen::Vector3d& v,
                                             const Eigen::VectorXd& x,
                                             const std::vector<int>& ids,
                                             const std::vector<double>& ws,
                                             int offset) const
{
    Eigen::Vector3d out = Eigen::Vector3d::Zero();

    // Defensive normalization
    double sw = 0.0; for (double w : ws) sw += w; const double inv_sw = (sw>1e-12)?1.0/sw:1.0;

    for (size_t k = 0; k < ids.size(); ++k) {
        const int nid = ids[k];
        const double wk = ws[k] * inv_sw;

        // Read node state from x
        Eigen::Matrix<double,6,1> se3 = x.segment<6>(offset + 6 * nid);
        Sophus::SE3d T = Sophus::SE3d::exp(se3);

        const Eigen::Vector3d& g = graph_[nid].position;
        out += wk * (T.so3() * (v - g) + g + T.translation());
    }
    return out;
}

void EDGraph::buildKnnNeighbors(int Ksmooth) {
    const int G = numNodes();
    node_neighbors_.assign(G, {});
    if (G <= 1) return;

    for (int i = 0; i < G; ++i) {
        std::vector<std::pair<int,double>> d; d.reserve(G-1);
        const Eigen::Vector3d gi = graph_[i].position;
        for (int j = 0; j < G; ++j) if (j != i) {
            double dist = (gi - graph_[j].position).norm();
            d.emplace_back(j, dist);
        }
        const int need = std::min(Ksmooth, (int)d.size());
        std::nth_element(d.begin(), d.begin()+need, d.end(),
                         [](const auto& a, const auto& b){ return a.second < b.second; });
        node_neighbors_[i].resize(need);
        for (int k = 0; k < need; ++k) node_neighbors_[i][k] = d[k].first;
    }
}

void EDGraph::updateFromStateVector(const Eigen::VectorXd& x, int offset) {
    const int G = numNodes();
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix<double,6,1> se3 = x.segment<6>(offset + 6 * i);
        graph_[i].transform = Sophus::SE3d::exp(se3);
    }
}

void EDGraph::writeToStateVector(Eigen::VectorXd& x, int offset) const {
    const int G = numNodes();
    if (x.size() < offset + 6*G) x.conservativeResize(offset + 6*G); // auto-resize for safety
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix<double,6,1> se3 = graph_[i].transform.log();
        x.segment<6>(offset + 6 * i) = se3;
    }
}