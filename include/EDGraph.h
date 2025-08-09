#ifndef EDGRAPH_H
#define EDGRAPH_H

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>
#include "MeshModel.h"  // 顶点定义

// 单个变形图节点结构
struct DeformationNode {
    Eigen::Vector3d position;       // 节点中心（rest position）
    Sophus::SE3d    transform;      // 当前 SE3 变换
    std::vector<int> neighbors;     // 邻接节点（用于平滑约束，可选）
};

class EDGraph {
public:
    explicit EDGraph(int K_bind = 4);

    // 用体素网格下采样从点云顶点构建节点，并立即对所有顶点做 KNN 绑定
    void initializeGraph(const std::vector<MeshModel::Vertex>& vertices,
                         double grid_size);

    // 直接设置节点（如外部构建），随后需手动 bindVertices
    void setGraphNodes(const std::vector<DeformationNode>& nodes);

    // 为每个网格顶点计算 K 近邻节点和权重
    void bindVertices(const std::vector<MeshModel::Vertex>& vertices);

    // 为每个节点建立 Ksmooth 邻接（用于后续平滑残差）
    void buildKnnNeighbors(int Ksmooth);

    // —— 形变 ——
    // 使用已绑定的节点/权重（按顶点索引）对单个顶点坐标做 ED 变形
    Eigen::Vector3d deformVertex(const Eigen::Vector3d& v, int vertex_index) const;

    // 在给定状态向量 x（每节点 6 维 se3）下，用明确的 node_ids & node_ws 对坐标 v 变形
    Eigen::Vector3d deformVertexByState(const Eigen::Vector3d& v,
                                        const Eigen::VectorXd& x,
                                        const std::vector<int>& node_ids,
                                        const std::vector<double>& node_ws,
                                        int offset) const;

    // —— 状态 I/O ——
    void updateFromStateVector(const Eigen::VectorXd& x, int offset);
    void writeToStateVector(Eigen::VectorXd& x, int offset) const; // 自动扩容

    // —— 访问器 ——
    int numNodes() const { return static_cast<int>(graph_.size()); }
    const std::vector<DeformationNode>& getGraphNodes() const { return graph_; }
    const std::vector<std::vector<int>>&    getBindings() const { return bindings_; }
    const std::vector<std::vector<double>>& getWeights()  const { return weights_; }

private:
    int K_;  // 每个顶点绑定的节点数量
    std::vector<DeformationNode> graph_;                  // 变形图节点
    std::vector<std::vector<int>> bindings_;              // 顶点 -> 节点索引（长度 K_ 或更少）
    std::vector<std::vector<double>> weights_;            // 顶点 -> 权重（与 bindings_ 一致）
};

#endif // EDGRAPH_H