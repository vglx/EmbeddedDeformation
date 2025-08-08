#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Dense>
#include "MeshModel.h"
#include "EDGraph.h"
#include "Optimizer.h"

// 手写解析 ASCII STL 文件
bool loadSTL_ASCII(const std::string& filename,
                   std::vector<Eigen::Vector3d>& vertices,
                   std::vector<MeshModel::Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open STL file: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::vector<Eigen::Vector3d> temp_vertices;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        if (prefix == "vertex") {
            double x, y, z;
            iss >> x >> y >> z;
            temp_vertices.emplace_back(x, y, z);
        }
    }
    file.close();

    if (temp_vertices.size() % 3 != 0) {
        std::cerr << "Invalid STL file: vertex count not divisible by 3." << std::endl;
        return false;
    }

    vertices = temp_vertices;
    triangles.clear();
    for (size_t i = 0; i < vertices.size(); i += 3) {
        MeshModel::Triangle t;
        t.v0 = static_cast<int>(i);
        t.v1 = static_cast<int>(i + 1);
        t.v2 = static_cast<int>(i + 2);
        triangles.push_back(t);
    }
    return true;
}

// 点云中心化
void centerPointCloud(std::vector<Eigen::Vector3d>& points) {
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const auto& p : points) center += p;
    center /= points.size();
    for (auto& p : points) p -= center;
}

// 采样关键点并高斯扰动
void pickUpKeypoints(const std::vector<Eigen::Vector3d>& pc,
                     double region_percent,
                     double max_deform,
                     std::vector<Eigen::Vector3d>& key_old,
                     std::vector<Eigen::Vector3d>& key_new,
                     std::vector<int>& key_indices) {
    Eigen::Vector3d min_pt = pc[0], max_pt = pc[0];
    for (const auto& p : pc) {
        min_pt = min_pt.cwiseMin(p);
        max_pt = max_pt.cwiseMax(p);
    }
    Eigen::Vector3d delta = max_pt - min_pt;
    Eigen::Vector3d region_min = min_pt + delta * ((1 - region_percent) / 2);
    Eigen::Vector3d region_max = max_pt - delta * ((1 - region_percent) / 2);

    key_old.clear();
    key_indices.clear();
    for (size_t i = 0; i < pc.size(); ++i) {
        const auto& p = pc[i];
        if ((p.array() >= region_min.array()).all() && (p.array() <= region_max.array()).all()) {
            key_old.push_back(p);
            key_indices.push_back(static_cast<int>(i));
        }
    }

    key_new = key_old;
    std::default_random_engine rng;
    std::normal_distribution<double> dist(0.0, max_deform / 3);
    for (auto& kp : key_new) {
        kp.x() += dist(rng);
        kp.y() += dist(rng);
        kp.z() += dist(rng);
    }
}

int main() {
    // 1. 加载 ASCII STL 模型
    std::string stl_file = "model_ascii.stl";
    std::vector<Eigen::Vector3d> original_points;
    std::vector<MeshModel::Triangle> triangles;
    if (!loadSTL_ASCII(stl_file, original_points, triangles)) {
        return -1;
    }

    // 2. 点云中心化
    centerPointCloud(original_points);

    // 3. 构造 MeshModel
    MeshModel model;
    std::vector<MeshModel::Vertex> mesh_vs;
    mesh_vs.reserve(original_points.size());
    for (const auto& p : original_points) {
        MeshModel::Vertex v; v.x = p.x(); v.y = p.y(); v.z = p.z();
        v.nx = v.ny = v.nz = 0.0f;
        mesh_vs.push_back(v);
    }
    model.setVertices(mesh_vs);
    model.setTriangles(triangles);
    model.computeVertexNormals();

    // 4. 采样关键点并扰动
    std::vector<Eigen::Vector3d> key_old, key_new;
    std::vector<int> key_indices;
    pickUpKeypoints(original_points, 0.4, 3.0, key_old, key_new, key_indices);

    // 5. 构建 EDGraph 控制节点
    std::vector<DeformationNode> nodes;
    int sampling_step = 50;
    const auto& verts = model.getVertices();
    for (size_t i = 0; i < verts.size(); i += sampling_step) {
        DeformationNode node;
        node.position = Eigen::Vector3d(verts[i].x, verts[i].y, verts[i].z);
        node.transform = Sophus::SE3d();
        nodes.push_back(node);
    }
    EDGraph edgraph(4);
    edgraph.setGraphNodes(nodes);
    edgraph.bindVertices(model.getVertices());

    // 6. 优化控制节点位姿
    Optimizer optimizer;
    Eigen::VectorXd x0;
    edgraph.writeToStateVector(x0, 0);
    Eigen::VectorXd x_opt;
    optimizer.optimize(x0, x_opt, edgraph, key_old, key_new, key_indices);
    edgraph.updateFromStateVector(x_opt, 0);

    // 7. 应用形变
    std::vector<Eigen::Vector3d> deformed;
    deformed.reserve(verts.size());
    for (size_t i = 0; i < verts.size(); ++i) {
        deformed.push_back(edgraph.deformVertex(verts[i], static_cast<int>(i)));
    }

    std::cout << "Loaded points: " << verts.size() << "  Deformed points: " << deformed.size() << std::endl;

    // 8. 保存变形前后点云为 PLY 文件
       auto savePLY = [](const std::string& filename, const std::vector<Eigen::Vector3d>& points) {
       std::ofstream ofs(filename);
       if (!ofs.is_open()) {
              std::cerr << "Failed to open file for writing: " << filename << std::endl;
              return;
       }
       ofs << "ply\nformat ascii 1.0\nelement vertex " << points.size()
              << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
       for (const auto& p : points) {
              ofs << p.x() << " " << p.y() << " " << p.z() << "\n";
       }
       ofs.close();
       std::cout << "Saved PLY: " << filename << std::endl;
       };

       savePLY("original.ply", original_points);
       savePLY("deformed.ply", deformed);


    return 0;
}