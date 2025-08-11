#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <random>
#include <algorithm>
#include <numeric>

#include <Eigen/Dense>

#include "MeshModel.h"
#include "EDGraph.h"
#include "Optimizer.h"

// -----------------------------
// Simple ASCII STL loader (NO dedup)
// -----------------------------
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
    temp_vertices.reserve(1<<18);

    while (std::getline(file, line)) {
        auto notspace = [](int ch){ return !std::isspace(ch); };
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), notspace));
        if (line.rfind("vertex", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double x, y, z; iss >> x >> y >> z;
            temp_vertices.emplace_back(x, y, z);
        }
    }
    file.close();

    if (temp_vertices.empty() || (temp_vertices.size() % 3 != 0)) {
        std::cerr << "Invalid ASCII STL: vertex count not divisible by 3." << std::endl;
        return false;
    }

    vertices = temp_vertices;             // no dedup, preserve order
    triangles.clear();
    triangles.reserve(vertices.size()/3);
    for (size_t i = 0; i + 2 < vertices.size(); i += 3) {
        MeshModel::Triangle t; t.v0 = (int)i; t.v1 = (int)(i+1); t.v2 = (int)(i+2);
        triangles.push_back(t);
    }
    return true;
}

// Load x y z per line
static std::vector<Eigen::Vector3d> loadXYZ(const std::string& path) {
    std::ifstream f(path);
    std::vector<Eigen::Vector3d> pts; pts.reserve(1<<16);
    double x,y,z;
    while (f >> x >> y >> z) pts.emplace_back(x,y,z);
    return pts;
}

// Map each key to nearest vertex index (for using that vertex's binding weights)
static std::vector<int> mapToNearestIndex(const std::vector<Eigen::Vector3d>& verts,
                                          const std::vector<Eigen::Vector3d>& keys) {
    std::vector<int> idx(keys.size(), -1);
    for (size_t i = 0; i < keys.size(); ++i) {
        double best = std::numeric_limits<double>::infinity();
        int best_id = -1;
        for (size_t v = 0; v < verts.size(); ++v) {
            double d = (verts[v] - keys[i]).squaredNorm();
            if (d < best) { best = d; best_id = (int)v; }
        }
        idx[i] = best_id;
    }
    return idx;
}

// Center the point cloud at origin (match MATLAB centering)
static void centerPointCloud(std::vector<Eigen::Vector3d>& P) {
    if (P.empty()) return;
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (auto& p: P) c += p; c /= (double)P.size();
    for (auto& p: P) p -= c;
}

// -------- Geometry comparison utilities (order-invariant) --------
static Eigen::MatrixXd toMat(const std::vector<Eigen::Vector3d>& pts) {
    Eigen::MatrixXd M(pts.size(), 3);
    for (size_t i = 0; i < pts.size(); ++i) M.row((int)i) = pts[i].transpose();
    return M;
}

// Kabsch (Procrustes without scaling/reflection): find R,t aligning B->A
static void kabschAlign(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                        Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    Eigen::Vector3d ca = A.colwise().mean();
    Eigen::Vector3d cb = B.colwise().mean();
    Eigen::MatrixXd Ac = A.rowwise() - ca.transpose();
    Eigen::MatrixXd Bc = B.rowwise() - cb.transpose();
    Eigen::Matrix3d H = Bc.transpose() * Ac;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    R = V * U.transpose();
    if (R.determinant() < 0) { V.col(2) *= -1; R = V * U.transpose(); }
    t = ca - R * cb;
}

// Brute-force NN distances from P to Q (each P finds nearest in Q). O(N*M)
static Eigen::VectorXd nnDists(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q) {
    const int N = (int)P.rows();
    const int M = (int)Q.rows();
    Eigen::VectorXd d(N);
    for (int i = 0; i < N; ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (int j = 0; j < M; ++j) {
            double dd = (P.row(i) - Q.row(j)).squaredNorm();
            if (dd < best) best = dd;
        }
        d[i] = std::sqrt(best);
    }
    return d;
}

static Eigen::MatrixXd subsampleRows(const Eigen::MatrixXd& X, int maxN) {
    if ((int)X.rows() <= maxN) return X;
    std::vector<int> idx(X.rows());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(123);
    std::shuffle(idx.begin(), idx.end(), rng);
    Eigen::MatrixXd Y(maxN, X.cols());
    for (int i = 0; i < maxN; ++i) Y.row(i) = X.row(idx[i]);
    return Y;
}

int main(int argc, char** argv) {
    std::string stl_file = (argc > 1 ? argv[1] : std::string("model_ascii.stl"));

    // 1) Load STL (no dedup)
    std::vector<Eigen::Vector3d> original_points;
    std::vector<MeshModel::Triangle> triangles;
    if (!loadSTL_ASCII(stl_file, original_points, triangles)) return -1;

    // 2) Center
    centerPointCloud(original_points);

    // 3) Build MeshModel
    MeshModel model;
    std::vector<MeshModel::Vertex> mesh_vs; mesh_vs.reserve(original_points.size());
    for (const auto& p : original_points) {
        MeshModel::Vertex v; v.x = p.x(); v.y = p.y(); v.z = p.z();
        v.nx = v.ny = v.nz = 0.0; mesh_vs.push_back(v);
    }
    model.setVertices(mesh_vs);
    model.setTriangles(triangles);

    // 4) Build ED graph via voxel downsampling (grid_size = 20) and bind vertices
    EDGraph edgraph(/*K_bind=*/6);
    edgraph.initializeGraph(model.getVertices(), /*grid_size=*/20.0);
    edgraph.buildKnnNeighbors(/*Ksmooth=*/5); // match MATLAB: num_nearestpts-1

    // 5) Load MATLAB keypoints and map to nearest vertex indices
    std::vector<Eigen::Vector3d> key_old = loadXYZ("../key_old.txt");
    std::vector<Eigen::Vector3d> key_new = loadXYZ("../key_new.txt");
    if (key_old.empty() || key_new.empty() || key_old.size() != key_new.size()) {
        std::cerr << "Keypoints invalid: key_old/new missing or size mismatch" << std::endl;
        return -1;
    }
    std::vector<int> key_indices = mapToNearestIndex(original_points, key_old);

    // 6) Initialize state and optimize (data + smooth)
    const int G = edgraph.numNodes();
    Eigen::VectorXd x0(6 * G);
    edgraph.writeToStateVector(x0, 0);

    Optimizer optimizer;
    Optimizer::Options opts; // tweak here if needed
    opts.max_iters = 50;
    opts.w_data    = 12.0;  // stronger data
    opts.w_smooth  = 1.0;   // gentle smoothness
    Eigen::VectorXd x_opt;
    optimizer.optimize(x0, x_opt, edgraph, key_old, key_new, key_indices, opts);
    edgraph.updateFromStateVector(x_opt, 0);

    // 7) Deform all vertices and dump
    std::vector<Eigen::Vector3d> deformed; deformed.reserve(mesh_vs.size());
    for (size_t i = 0; i < mesh_vs.size(); ++i) {
        const auto& mv = mesh_vs[i];
        deformed.push_back(edgraph.deformVertex(mv, (int)i));
    }

    // Save for MATLAB comparison
    {
        std::ofstream fout("../deformed_cpp.txt");
        for (const auto& p : deformed) fout << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
        std::cout << "Saved deformed_cpp.txt (" << deformed.size() << " points)\n";
    }

    // 8) If MATLAB output exists, do in-program geometric comparison
    std::ifstream test("../deformed_matlab.txt");
    if (test.good()) {
        std::cout << "\n[Compare] Found deformed_matlab.txt — running alignment & metrics...\n";
        std::vector<Eigen::Vector3d> matlab_pts = loadXYZ("deformed_matlab.txt");
        if (matlab_pts.empty()) {
            std::cerr << "[Compare] deformed_matlab.txt is empty or unreadable.\n";
            return 0;
        }
        Eigen::MatrixXd A = toMat(matlab_pts);
        Eigen::MatrixXd B = toMat(deformed);

        const int maxN = 120000; // adjust if needed
        A = subsampleRows(A, maxN);
        B = subsampleRows(B, maxN);

        Eigen::Matrix3d R; Eigen::Vector3d t;
        kabschAlign(A, B, R, t);
        Eigen::MatrixXd B_aligned = (B * R.transpose()).rowwise() + t.transpose();

        Eigen::VectorXd dA = nnDists(A, B_aligned);
        Eigen::VectorXd dB = nnDists(B_aligned, A);

        double rmse    = std::sqrt(dA.array().square().mean());
        double chamfer = dA.mean() + dB.mean();
        double haus    = std::max(dA.maxCoeff(), dB.maxCoeff());

        std::cout << "RMSE (NN-correspondence, after alignment): " << rmse << "\n";
        std::cout << "Chamfer distance (symmetric mean NN): " << chamfer
                  << "  [meanA=" << dA.mean() << ", meanB=" << dB.mean() << "]\n";
        std::cout << "Hausdorff distance: " << haus << "\n";
    } else {
        std::cout << "\n[Compare] deformed_matlab.txt not found — skip metrics.\n";
    }

    return 0;
}