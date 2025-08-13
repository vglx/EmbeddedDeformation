#include "EDGraph.h"
#include <algorithm>
#include <limits>
#include <unordered_map>

// ---- Voxel hash key for downsampling ----
struct VoxelKey { int x, y, z; };
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const noexcept {
        std::size_t h = 1469598103934665603ull;
        h ^= std::hash<int>{}(k.x) + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.y) + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
        h ^= std::hash<int>{}(k.z) + 0x9e3779b97f4a7c15ull + (h<<6) + (h>>2);
        return h;
    }
};
struct VoxelKeyEq { bool operator()(const VoxelKey& a, const VoxelKey& b) const noexcept { return a.x==b.x && a.y==b.y && a.z==b.z; } };

EDGraph::EDGraph(int K) : K_(K) {}

void EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& vertices, double grid_size) {
    std::vector<DeformationNode> nodes;
    voxelDownsample(vertices, grid_size, nodes);
    setGraphNodes(nodes);
    bindVertices(vertices);
}

void EDGraph::setGraphNodes(const std::vector<DeformationNode>& nodes) { graph_ = nodes; }

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
        int ix = static_cast<int>(std::floor(p.x() / gs));
        int iy = static_cast<int>(std::floor(p.y() / gs));
        int iz = static_cast<int>(std::floor(p.z() / gs));
        return VoxelKey{ix,iy,iz};
    };

    for (const auto& v : vertices) {
        Eigen::Vector3d p(v.x, v.y, v.z);
        VoxelKey key = toKey(p);
        auto it = buckets.find(key);
        if (it == buckets.end()) buckets.emplace(key, Accum{p, 1});
        else { it->second.sum += p; it->second.cnt += 1; }
    }

    out_nodes.reserve(buckets.size());
    for (const auto& kv : buckets) {
        DeformationNode node;
        node.position = kv.second.sum / std::max(1, kv.second.cnt);
        node.A.setIdentity();
        node.t.setZero();
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

        // distances to all nodes
        std::vector<std::pair<int,double>> dists; dists.reserve(G);
        for (int j = 0; j < G; ++j) {
            double dist = (v - graph_[j].position).norm();
            dists.emplace_back(j, dist);
        }

        // === MATLAB-equivalent base distance: use (K+1)-th neighbor when available ===
        const int kth_bind = std::min(K_, (int)dists.size());        // we will keep K_ nearest for binding
        const int base_rank = std::min(K_, std::max(0, G - 1));      // index of (K+1)-th (0-based) if exists
        std::nth_element(dists.begin(), dists.begin() + base_rank, dists.end(),
                         [](const auto& a, const auto& b){ return a.second < b.second; });
        double base = dists[base_rank].second;                        // base distance (K+1-th)
        if (base < 1e-12) base = 1e-12;

        // partially sort top-K for actual binding
        std::partial_sort(dists.begin(), dists.begin()+kth_bind, dists.end(),
                          [](const auto& a, const auto& b){ return a.second < b.second; });

        bindings_[i].resize(kth_bind);
        weights_[i].resize(kth_bind);
        double sumw = 0.0;
        for (int k = 0; k < kth_bind; ++k) {
            bindings_[i][k] = dists[k].first;
            const double raw = 1.0 - dists[k].second / base;          // 1 - d / d_{K+1}
            const double w   = std::max(0.0, raw);
            weights_[i][k]   = w;
            sumw += w;
        }
        // Row-normalize (MATLAB UpdateWeights does this)
        if (sumw < 1e-12) {
            const double uni = 1.0 / std::max(1, kth_bind);
            for (double& w : weights_[i]) w = uni;
        } else {
            for (double& w : weights_[i]) w /= sumw;
        }
    }
}

Eigen::Vector3d EDGraph::deformVertex(const MeshModel::Vertex& vertex, int vidx) const {
    const Eigen::Vector3d v(vertex.x, vertex.y, vertex.z);
    return deformVertex(v, vidx);
}

Eigen::Vector3d EDGraph::deformVertex(const Eigen::Vector3d& v, int vidx) const {
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    const auto& ids = bindings_[vidx];
    const auto& ws  = weights_[vidx];
    double sw = 0.0; for (double w : ws) sw += w; const double inv_sw = (sw>1e-12)?1.0/sw:1.0;
    for (size_t k = 0; k < ids.size(); ++k) {
        const auto& node = graph_[ids[k]];
        const Eigen::Vector3d& g = node.position;
        out += (ws[k]*inv_sw) * (node.A * (v - g) + g + node.t);
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
    double sw = 0.0; for (double w : ws) sw += w; const double inv_sw = (sw>1e-12)?1.0/sw:1.0;
    for (size_t k = 0; k < ids.size(); ++k) {
        const int nid = ids[k];
        const double wk = ws[k] * inv_sw;
        const int base = offset + 12 * nid;
        Eigen::Matrix3d A;
        A << x(base+0), x(base+1), x(base+2),
             x(base+3), x(base+4), x(base+5),
             x(base+6), x(base+7), x(base+8);
        Eigen::Vector3d t(x(base+9), x(base+10), x(base+11));
        const Eigen::Vector3d& g = graph_[nid].position;
        out += wk * (A * (v - g) + g + t);
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
        const int base = offset + 12 * i;
        Eigen::Matrix3d A;
        A << x(base+0), x(base+1), x(base+2),
             x(base+3), x(base+4), x(base+5),
             x(base+6), x(base+7), x(base+8);
        Eigen::Vector3d t(x(base+9), x(base+10), x(base+11));
        graph_[i].A = A; graph_[i].t = t;
    }
}

void EDGraph::writeToStateVector(Eigen::VectorXd& x, int offset) const {
    const int G = numNodes();
    const int need = offset + 12 * G;
    if (x.size() < need) x.conservativeResize(need);

    for (int i = 0; i < G; ++i) {
        const int base = offset + 12 * i;
        const Eigen::Matrix3d& A = graph_[i].A;
        const Eigen::Vector3d& t = graph_[i].t;
        x(base+0) = A(0,0); x(base+1) = A(0,1); x(base+2) = A(0,2);
        x(base+3) = A(1,0); x(base+4) = A(1,1); x(base+5) = A(1,2);
        x(base+6) = A(2,0); x(base+7) = A(2,1); x(base+8) = A(2,2);
        x(base+9) = t(0);   x(base+10)= t(1);   x(base+11)= t(2);
    }
}