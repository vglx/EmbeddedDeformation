// main.cpp (modified for grid sampling, Gaussian weights, and iterative embedding deformation)
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

// Load ASCII STL file into vertices and triangles
bool loadSTL_ASCII(const std::string& filename,
                   std::vector<Eigen::Vector3d>& vertices,
                   std::vector<MeshModel::Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open STL file: " << filename << std::endl;
        return false;
    }
    std::string line;
    std::vector<Eigen::Vector3d> temp;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        if (prefix == "vertex") {
            double x, y, z;
            iss >> x >> y >> z;
            temp.emplace_back(x, y, z);
        }
    }
    file.close();
    if (temp.size() % 3 != 0) {
        std::cerr << "Invalid STL: vertex count not divisible by 3." << std::endl;
        return false;
    }
    vertices = temp;
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

// Center point cloud at origin
void centerPointCloud(std::vector<Eigen::Vector3d>& pts) {
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (auto& p : pts) c += p;
    c /= pts.size();
    for (auto& p : pts) p -= c;
}

void pickUpKeypoints(const std::vector<Eigen::Vector3d>& pc,
                     double region_percent,
                     double max_deform,
                     std::vector<Eigen::Vector3d>& key_old,
                     std::vector<Eigen::Vector3d>& key_new) {
    Eigen::Vector3d min_pt = pc[0], max_pt = pc[0];
    for (const auto& p : pc) {
        min_pt = min_pt.cwiseMin(p);
        max_pt = max_pt.cwiseMax(p);
    }
    Eigen::Vector3d delta = max_pt - min_pt;
    Eigen::Vector3d region_min = min_pt + delta * ((1 - region_percent) / 2);
    Eigen::Vector3d region_max = max_pt - delta * ((1 - region_percent) / 2);

    key_old.resize(6);
    key_new.resize(6);
    for (int i = 0; i < 6; ++i) key_old[i] = pc[0];
    for (const auto& p : pc) {
        if ((p.array() >= region_min.array()).all() &&
            (p.array() <= region_max.array()).all()) {
            if (p.z() > key_old[0].z()) key_old[0] = p;
            if (p.z() < key_old[1].z()) key_old[1] = p;
            if (p.y() > key_old[2].y()) key_old[2] = p;
            if (p.y() < key_old[3].y()) key_old[3] = p;
            if (p.x() > key_old[4].x()) key_old[4] = p;
            if (p.x() < key_old[5].x()) key_old[5] = p;
        }
    }

    key_new = key_old;
    std::default_random_engine rng;
    std::normal_distribution<double> dist(0.0, max_deform / 3);
    key_new[0].z() += std::abs(dist(rng));
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
    pickUpKeypoints(original_points, 0.4, 3.0, key_old, key_new);

    // 5. 构建 EDGraph 控制节点
    int sampling_step = 50;
    const auto& verts = model.getVertices();
    EDGraph edgraph(6);  // K = 6
    edgraph.initializeGraph(verts, 20);  // grid_size = 20

    // 6. 优化控制节点位姿
    Optimizer optimizer;
    Eigen::VectorXd x0;
    edgraph.writeToStateVector(x0, 0);
    Eigen::VectorXd x_opt;
    optimizer.optimize(x0, x_opt, edgraph, key_old, key_new);
    edgraph.updateFromStateVector(x_opt, 0);

    // 7. 应用形变并输出
    std::vector<Eigen::Vector3d> deformed;
    deformed.reserve(verts.size());
    for (size_t i = 0; i < verts.size(); ++i) {
        deformed.push_back(edgraph.deformVertex(verts[i], static_cast<int>(i)));
    }

    std::ofstream fout("deformed_cpp.txt");
    for (const auto& p : deformed) fout << p.transpose() << "\n";
    std::cout << "Saved deformed point cloud to deformed_cpp.txt\n";

    return 0;
}