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

// Sample 6 extremal keypoints and apply Gaussian perturbation
void pickUpKeypoints(const std::vector<Eigen::Vector3d>& pc,
                     double region_percent,
                     double max_deform,
                     std::vector<Eigen::Vector3d>& key_old,
                     std::vector<Eigen::Vector3d>& key_new,
                     std::vector<int>& key_indices) {
    // compute region bounds
    Eigen::Vector3d min_pt = pc[0], max_pt = pc[0];
    for (auto& p : pc) {
        min_pt = min_pt.cwiseMin(p);
        max_pt = max_pt.cwiseMax(p);
    }
    Eigen::Vector3d delta = max_pt - min_pt;
    Eigen::Vector3d region_min = min_pt + delta * ((1 - region_percent) / 2);
    Eigen::Vector3d region_max = max_pt - delta * ((1 - region_percent) / 2);

    key_old.assign(6, pc[0]);
    key_new.assign(6, pc[0]);
    key_indices.assign(6, 0);
    // find extremal points in region
    for (int i = 0; i < (int)pc.size(); ++i) {
        const auto& p = pc[i];
        if ((p.array() >= region_min.array()).all() && (p.array() <= region_max.array()).all()) {
            if (p.z() > key_old[0].z()) { key_old[0] = p; key_indices[0] = i; }
            if (p.z() < key_old[1].z()) { key_old[1] = p; key_indices[1] = i; }
            if (p.y() > key_old[2].y()) { key_old[2] = p; key_indices[2] = i; }
            if (p.y() < key_old[3].y()) { key_old[3] = p; key_indices[3] = i; }
            if (p.x() > key_old[4].x()) { key_old[4] = p; key_indices[4] = i; }
            if (p.x() < key_old[5].x()) { key_old[5] = p; key_indices[5] = i; }
        }
    }
    // Gaussian perturbation
    std::default_random_engine rng;
    std::normal_distribution<double> dist(0.0, max_deform / 3);
    for (int k = 0; k < 6; ++k) {
        key_new[k] = key_old[k];
        key_new[k].x() += dist(rng);
        key_new[k].y() += dist(rng);
        key_new[k].z() += dist(rng);
    }
}

int main() {
    // 1. Load and center
    std::string stl_file = "model_ascii.stl";
    std::vector<Eigen::Vector3d> original_points;
    std::vector<MeshModel::Triangle> triangles;
    if (!loadSTL_ASCII(stl_file, original_points, triangles)) return -1;
    centerPointCloud(original_points);

    // 2. Build MeshModel
    MeshModel model;
    std::vector<MeshModel::Vertex> mesh_vs;
    mesh_vs.reserve(original_points.size());
    for (auto& p : original_points) {
        MeshModel::Vertex v;
        v.x = p.x(); v.y = p.y(); v.z = p.z();
        v.nx = v.ny = v.nz = 0.0f;
        mesh_vs.push_back(v);
    }
    model.setVertices(mesh_vs);
    model.setTriangles(triangles);
    model.computeVertexNormals();

    // 3. Initialize EDGraph with grid sampling
    int grid_size = 20;
    EDGraph edgraph(6);
    edgraph.initializeGraph(model.getVertices(), grid_size);

    // 4. Iterative embedding deformation
    std::vector<Eigen::Vector3d> current_points = original_points;
    Optimizer optimizer;
    std::vector<Eigen::Vector3d> key_old, key_new;
    std::vector<int> key_indices;
    int outer_iter = 3;
    double region_percent = 0.4;
    double max_deform = 3.0;
    for (int epoch = 0; epoch < outer_iter; ++epoch) {
        pickUpKeypoints(current_points, region_percent, max_deform,
                        key_old, key_new, key_indices);
        Eigen::VectorXd x0(6 * edgraph.numNodes()), x_opt;
        edgraph.writeToStateVector(x0, 0);
        optimizer.optimize(x0, x_opt, edgraph,
                           key_old, key_new, key_indices);
        edgraph.updateFromStateVector(x_opt, 0);
        // apply deformation to all points
        for (int i = 0; i < current_points.size(); ++i) {
            MeshModel::Vertex v;
            v.x = static_cast<float>(current_points[i].x());
            v.y = static_cast<float>(current_points[i].y());
            v.z = static_cast<float>(current_points[i].z());
            v.nx = v.ny = v.nz = 0.0f;
            current_points[i] = edgraph.deformVertex(v, i);
        }
    }

    // 5. Save final deformed result as TXT
    auto saveTXT = [](const std::string& filename,
                      const std::vector<Eigen::Vector3d>& pts) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }
        for (auto& p : pts)
            ofs << p.x() << " " << p.y() << " " << p.z() << "\n";
        ofs.close();
        std::cout << "Saved TXT: " << filename << std::endl;
    };
    saveTXT("deformed_cpp.txt", current_points);

    return 0;
}