// main.cpp — patched version
// - Safely initializes state vector x0 (6 * numNodes)
// - Maps keypoints to vertex indices (key_indices) and passes them to Optimizer
// - Minimal ASCII STL loader
// - Writes deformed point cloud to deformed_cpp.txt

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <limits>
#include <cmath>

#include <Eigen/Dense>

#include "MeshModel.h"
#include "EDGraph.h"
#include "Optimizer.h"

// -----------------------------
// Simple ASCII STL loader
// -----------------------------
// Loads vertices (unique-ified) and triangles from an ASCII STL file.
// Note: ASCII STL lists each triangle's 3 vertices; we deduplicate by position.
static bool loadSTL_ASCII(
    const std::string& filename,
    std::vector<Eigen::Vector3d>& out_vertices,
    std::vector<MeshModel::Triangle>& out_tris)
{
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Failed to open STL: " << filename << "\n";
        return false;
    }

    // Map (x,y,z) -> index by string key (simple but sufficient)
    std::vector<Eigen::Vector3d> raw;
    raw.reserve(1 << 18);

    std::string line;
    std::vector<int> tri_indices;
    tri_indices.reserve(1 << 18);

    // We will deduplicate after reading
    while (std::getline(fin, line)) {
        // trim head
        auto notspace = [](int ch){ return !std::isspace(ch); };
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), notspace));
        if (line.rfind("vertex", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double x, y, z; iss >> x >> y >> z;
            raw.emplace_back(x, y, z);
        }
    }

    if (raw.empty() || raw.size() % 3 != 0) {
        std::cerr << "STL appears empty or not multiple of 3 vertices (triangles)." << std::endl;
        return false;
    }

    // Deduplicate using a simple linear search with tolerance (could be optimized with hash)
    const double eps2 = 1e-18; // exact compare by double is usually okay for ASCII STL; keep eps just in case
    for (size_t i = 0; i < raw.size(); ++i) {
        const Eigen::Vector3d& v = raw[i];
        int found = -1;
        for (size_t j = 0; j < out_vertices.size(); ++j) {
            if ((out_vertices[j] - v).squaredNorm() <= eps2) { found = static_cast<int>(j); break; }
        }
        if (found < 0) {
            out_vertices.push_back(v);
            tri_indices.push_back(static_cast<int>(out_vertices.size() - 1));
        } else {
            tri_indices.push_back(found);
        }
    }

    // Build triangles
    out_tris.clear();
    out_tris.reserve(tri_indices.size() / 3);
    for (size_t t = 0; t + 2 < tri_indices.size(); t += 3) {
        MeshModel::Triangle tri;
        tri.v0 = tri_indices[t + 0];
        tri.v1 = tri_indices[t + 1];
        tri.v2 = tri_indices[t + 2];
        out_tris.push_back(tri);
    }

    return true;
}

// Map each key point (world position) to the nearest vertex index in verts
static std::vector<int> mapToNearestIndex(
    const std::vector<Eigen::Vector3d>& verts,
    const std::vector<Eigen::Vector3d>& keys)
{
    std::vector<int> idx(keys.size(), -1);
    for (size_t i = 0; i < keys.size(); ++i) {
        double best = std::numeric_limits<double>::infinity();
        int best_id = -1;
        for (size_t v = 0; v < verts.size(); ++v) {
            double d = (verts[v] - keys[i]).squaredNorm();
            if (d < best) { best = d; best_id = static_cast<int>(v); }
        }
        idx[i] = best_id;
    }
    return idx;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_ascii.stl>\n";
        return 0;
    }

    const std::string stl_path = argv[1];

    // 1) Load STL (ASCII)
    std::vector<Eigen::Vector3d> verts;               // unique positions
    std::vector<MeshModel::Triangle> tris;            // triangle indices
    if (!loadSTL_ASCII(stl_path, verts, tris)) return -1;
    std::cout << "Loaded vertices: " << verts.size() << ", triangles: " << tris.size() << "\n";

    // 2) Center the model at origin (optional but keeps parity with MATLAB scripts)
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (auto& v : verts) c += v;
    if (!verts.empty()) c /= double(verts.size());
    for (auto& v : verts) v -= c;

    // 3) Fill MeshModel for EDGraph binding
    std::vector<MeshModel::Vertex> mesh_verts;
    mesh_verts.reserve(verts.size());
    for (const auto& v : verts) {
        MeshModel::Vertex mv; mv.x = float(v.x()); mv.y = float(v.y()); mv.z = float(v.z());
        mv.nx = mv.ny = mv.nz = 0.f;
        mesh_verts.push_back(mv);
    }

    MeshModel mesh;
    mesh.setVertices(mesh_verts);
    mesh.setTriangles(tris);

    // 4) Build ED graph (voxel grid downsampling -> nodes) and bind vertices (K=6)
    EDGraph edgraph(/*K_bind=*/6);
    edgraph.initializeGraph(mesh.getVertices(), /*grid_size=*/20.0);
    edgraph.buildKnnNeighbors(6); 

    // 5) Prepare a small set of key constraints (demo): pick N random vertices and move them slightly
    //    In your pipeline, you should load key_old/key_new from MATLAB txt to align exactly.
    const int Nkeys = 6; // keep same as MATLAB pickUpPoints default (6)
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> unif(0, int(verts.size()) - 1);

    std::vector<Eigen::Vector3d> key_old, key_new;
    key_old.reserve(Nkeys); key_new.reserve(Nkeys);

    for (int i = 0; i < Nkeys; ++i) {
        int vid = unif(rng);
        Eigen::Vector3d vo = verts[vid];
        key_old.push_back(vo);
        // small positive z-perturbation (similar spirit to MATLAB pickUpPoints first key)
        Eigen::Vector3d vn = vo; vn.z() += 3.0; // max_deform ~ 3 (example)
        key_new.push_back(vn);
    }

    // 6) Map keys to nearest vertex indices (critical for using precomputed bindings/weights)
    std::vector<int> key_indices = mapToNearestIndex(verts, key_old);

    // 7) Initialize state x0 with correct size (6 * numNodes) and write initial transforms
    const int G = edgraph.numNodes();
    Eigen::VectorXd x0(6 * G);
    edgraph.writeToStateVector(x0, /*offset=*/0);

    // 8) Optimize node transforms so that ED(key_old) ≈ key_new
    Optimizer opt;
    Eigen::VectorXd x_opt;
    opt.optimize(x0, x_opt, edgraph, key_old, key_new, key_indices);

    // 9) Update graph with optimized state
    edgraph.updateFromStateVector(x_opt, /*offset=*/0);

    // 10) Apply deformation to all vertices and dump to txt (for RMSE check against MATLAB)
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