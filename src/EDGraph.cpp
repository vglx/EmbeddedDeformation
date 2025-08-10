#include "EDGraph.h"
#include <algorithm>
#include <cmath>

EDGraph::EDGraph(int K): K_(K) {}

void EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& vertices, int sampling_step) {
    std::vector<DeformationNode> nodes;
    nodes.reserve(vertices.size() / std::max(1, sampling_step) + 1);

    for (size_t i = 0; i < vertices.size(); i += std::max(1, sampling_step)) {
        const auto& v = vertices[i];
        DeformationNode node;
        node.position  = Eigen::Vector3d(v.x, v.y, v.z);
        node.transform = Sophus::SE3d();  // identity
        nodes.push_back(node);
    }
    setGraphNodes(nodes);
    bindVertices(vertices);
}

void EDGraph::setGraphNodes(const std::vector<DeformationNode>& nodes) {
    graph_ = nodes;
}

void EDGraph::bindVertices(const std::vector<MeshModel::Vertex>& vertices) {
    const size_t nV = vertices.size();
    const int    G  = static_cast<int>(graph_.size());
    bindings_.assign(nV, {});
    weights_.assign(nV, {});
    if (G == 0) return;

    for (size_t i = 0; i < nV; ++i) {
        const Eigen::Vector3d v(vertices[i].x, vertices[i].y, vertices[i].z);

        // Gather distances to all nodes
        std::vector<std::pair<int,double>> dists; // (node_id, Euclidean distance)
        dists.reserve(G);
        for (int j = 0; j < G; ++j) {
            double dist = (v - graph_[j].position).norm();
            dists.emplace_back(j, dist);
        }

        // Take K+1 nearest to form the base distance (MATLAB-style), then keep first K for weights
        const int need = std::min(K_ + 1, static_cast<int>(dists.size()));
        std::nth_element(dists.begin(), dists.begin() + need, dists.end(),
                         [](const auto& a, const auto& b){ return a.second < b.second; });

        // Base distance is the (K+1)-th nearest (protect from zero)
        double base = dists[need - 1].second;
        if (base < 1e-12) base = 1e-12;

        const int kth = std::min(K_, static_cast<int>(dists.size()));
        bindings_[i].resize(kth);
        weights_[i].resize(kth);

        double sumw = 0.0;
        for (int k = 0; k < kth; ++k) {
            bindings_[i][k] = dists[k].first;
            double raw = 1.0 - dists[k].second / base;   // in [0,1] ideally
            double w   = std::max(0.0, raw);             // clamp negatives
            weights_[i][k] = w;
            sumw += w;
        }

        // Normalize (fallback to uniform if degenerate)
        if (sumw < 1e-12) {
            for (int k = 0; k < kth; ++k) weights_[i][k] = 1.0 / std::max(1, kth);
        } else {
            for (int k = 0; k < kth; ++k) weights_[i][k] /= sumw;
        }
    }
}

Eigen::Vector3d EDGraph::deformVertex(const MeshModel::Vertex& vertex, int vidx) const {
    const Eigen::Vector3d v(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d out = Eigen::Vector3d::Zero();

    const auto& ids = bindings_[vidx];
    const auto& ws  = weights_[vidx];
    for (size_t k = 0; k < ids.size(); ++k) {
        const auto& node = graph_[ids[k]];
        const Eigen::Vector3d& g = node.position;
        out += ws[k] * (node.transform.so3() * (v - g) + g + node.transform.translation());
    }
    return out;
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