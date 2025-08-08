#include "EDGraph.h"
#include <algorithm>

EDGraph::EDGraph(int K): K_(K) {}

void EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& vertices, int sampling_step) {
    std::vector<DeformationNode> nodes;
    nodes.reserve(vertices.size() / sampling_step + 1);

    for (size_t i = 0; i < vertices.size(); i += sampling_step) {
        const auto& v = vertices[i];
        DeformationNode node;
        node.position = Eigen::Vector3d(v.x, v.y, v.z);
        node.transform = Sophus::SE3d();  // 默认单位变换
        nodes.push_back(node);
    }
    setGraphNodes(nodes);           // 设置类内部 nodes_
    bindVertices(vertices);    // 绑定顶点 -> 控制点
}

void EDGraph::setGraphNodes(const std::vector<DeformationNode>& nodes) {
    graph_ = nodes;
}

void EDGraph::bindVertices(const std::vector<MeshModel::Vertex>& vertices) {
    size_t n = vertices.size();
    bindings_.assign(n, {});
    weights_.assign(n, {});

    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3d v(vertices[i].x, vertices[i].y, vertices[i].z);
        std::vector<std::pair<int, double>> dists;
        dists.reserve(graph_.size());
        for (int j = 0; j < (int)graph_.size(); ++j) {
            double dist = (v - graph_[j].position).norm();
            dists.emplace_back(j, dist);
        }
        std::sort(dists.begin(), dists.end(), [](auto& a, auto& b) {
            return a.second < b.second;
        });

        double sum = 0.0;
        int kth = std::min(K_, (int)dists.size());
        bindings_[i].resize(kth);
        weights_[i].resize(kth);
        for (int k = 0; k < kth; ++k) {
            bindings_[i][k] = dists[k].first;
            double w = 1.0 / (dists[k].second + 1e-8);
            weights_[i][k] = w;
            sum += w;
        }
        for (double& w : weights_[i]) w /= sum;
    }
}

Eigen::Vector3d EDGraph::deformVertex(const MeshModel::Vertex& vertex, int vidx) const {
    Eigen::Vector3d v(vertex.x, vertex.y, vertex.z);
    Eigen::Vector3d result(0, 0, 0);
    const auto& node_ids = bindings_[vidx];
    const auto& node_ws = weights_[vidx];
    for (size_t k = 0; k < node_ids.size(); ++k) {
        int nid = node_ids[k];
        double w = node_ws[k];
        const auto& node = graph_[nid];
        Eigen::Vector3d g = node.position;
        // ED 变形公式：R_j (v - g_j) + g_j + t_j
        Eigen::Vector3d p = node.transform.so3() * (v - g) + g + node.transform.translation();
        result += w * p;
    }
    return result;
}

Eigen::Vector3d EDGraph::deformVertexByState(const Eigen::Vector3d& v,
                                             const Eigen::VectorXd& x,
                                             int vidx,
                                             int offset) const {
    Eigen::Vector3d result(0, 0, 0);
    const auto& node_ids = bindings_[vidx];
    const auto& node_ws = weights_[vidx];
    for (size_t k = 0; k < node_ids.size(); ++k) {
        int nid = node_ids[k];
        double w = node_ws[k];
        Eigen::Matrix<double,6,1> se3 = x.segment<6>(offset + 6 * nid);
        Sophus::SE3d T = Sophus::SE3d::exp(se3);
        Eigen::Vector3d g = graph_[nid].position;
        Eigen::Vector3d p = T.so3() * (v - g) + g + T.translation();
        result += w * p;
    }
    return result;
}

void EDGraph::updateFromStateVector(const Eigen::VectorXd& x, int offset) {
    int G = numNodes();
    for (int i = 0; i < G; ++i) {
        // 每个节点 6 个自由度
        Eigen::Matrix<double,6,1> se3 = x.segment<6>(offset + 6 * i);
        graph_[i].transform = Sophus::SE3d::exp(se3);
    }
}

void EDGraph::writeToStateVector(Eigen::VectorXd& x, int offset) const {
    int G = numNodes();
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix<double,6,1> se3 = graph_[i].transform.log();
        x.segment<6>(offset + 6 * i) = se3;
    }
}