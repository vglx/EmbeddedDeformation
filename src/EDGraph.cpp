#include "EDGraph.h"
#include <algorithm>

EDGraph::EDGraph(int K): K_(K) {}

void EDGraph::initializeGraph(const std::vector<MeshModel::Vertex>& vertices, int grid_size) {
    // uniformly sample on a grid in bounding box
    Eigen::Vector3d min_pt(vertices[0].x, vertices[0].y, vertices[0].z);
    Eigen::Vector3d max_pt = min_pt;
    for (auto& v: vertices) {
        Eigen::Vector3d p(v.x, v.y, v.z);
        min_pt = min_pt.cwiseMin(p);
        max_pt = max_pt.cwiseMax(p);
    }
    Eigen::Vector3d extent = max_pt - min_pt;
    int nx = std::max(1, int(extent.x() / grid_size));
    int ny = std::max(1, int(extent.y() / grid_size));
    int nz = std::max(1, int(extent.z() / grid_size));
    graph_.clear();
    for (int i = 0; i <= nx; ++i)
    for (int j = 0; j <= ny; ++j)
    for (int k = 0; k <= nz; ++k) {
        Eigen::Vector3d pos = min_pt + Eigen::Vector3d(i * grid_size, j * grid_size, k * grid_size);
        graph_.push_back({pos, Sophus::SE3d()});
    }
    bindVertices(vertices);
}

void EDGraph::setGraphNodes(const std::vector<DeformationNode>& nodes) {
    graph_ = nodes;
}

void EDGraph::bindVertices(const std::vector<MeshModel::Vertex>& vertices) {
    int n = vertices.size();
    bindings_.assign(n, {});
    weights_.assign(n, {});
    double sigma = 10.0; // e.g. grid_size/2
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d p(vertices[i].x, vertices[i].y, vertices[i].z);
        std::vector<std::pair<int,double>> dist_idx;
        for (int j = 0; j < graph_.size(); ++j) {
            double d = (p - graph_[j].position).norm();
            dist_idx.emplace_back(j, d);
        }
        std::sort(dist_idx.begin(), dist_idx.end(), [](auto&a,auto&b){return a.second<b.second;});
        int kth = std::min(K_, (int)dist_idx.size());
        bindings_[i].resize(kth);
        weights_[i].resize(kth);
        double sum=0;
        for(int k=0;k<kth;++k){
            bindings_[i][k] = dist_idx[k].first;
            double w = std::exp(-dist_idx[k].second*dist_idx[k].second/(2*sigma*sigma));
            weights_[i][k] = w;
            sum += w;
        }
        for(auto &w:weights_[i]) w/=sum;
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