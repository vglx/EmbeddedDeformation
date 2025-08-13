#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <filesystem>

#include <Eigen/Dense>

#include "MeshModel.h"
#include "EDGraph.h"
#include "Optimizer.h"

using Vec3 = Eigen::Vector3d;
namespace fs = std::filesystem;

// -----------------------------
// Utilities
// -----------------------------
static std::vector<Vec3> loadXYZ(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) { return {}; }
    std::vector<Vec3> pts; pts.reserve(1<<16);
    double x,y,z; while (f >> x >> y >> z) pts.emplace_back(x,y,z);
    return pts;
}

static std::vector<int> loadIndices1Based(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) { return {}; }
    std::vector<int> idx; idx.reserve(1024);
    long long v; while (f >> v) idx.push_back(static_cast<int>(v - 1));
    return idx;
}

static bool loadInt(const std::string& path, int& out_val) {
    std::ifstream f(path); if (!f.is_open()) return false; f >> out_val; return true;
}

static bool saveVector(const std::string& path, const Eigen::VectorXd& x){
    std::ofstream f(path); if(!f.is_open()) return false;
    for(int i=0;i<x.size();++i){ f << std::setprecision(17) << x[i] << '\n'; }
    return true;
}

static std::vector<Vec3> deformAll(const EDGraph& ed,
                                   const std::vector<MeshModel::Vertex>& V,
                                   const Eigen::VectorXd& x) {
    const auto& B = ed.getVertexBindings();
    const auto& W = ed.getVertexWeights();
    std::vector<Vec3> out; out.reserve(V.size());
    for (size_t i = 0; i < V.size(); ++i) {
        Vec3 p(V[i].x, V[i].y, V[i].z);
        out.push_back(ed.deformVertexByState(p, x, B[i], W[i], 0));
    }
    return out;
}

static void keyErrors(const EDGraph& ed,
                      const std::vector<Vec3>& key_old,
                      const std::vector<Vec3>& key_new,
                      const std::vector<int>& key_idx,
                      const Eigen::VectorXd& x,
                      double& mean_e, double& rmse_e, double& max_e) {
    const auto& B = ed.getVertexBindings();
    const auto& W = ed.getVertexWeights();
    double sum=0, sum2=0, mx=0; int N = (int)key_old.size();
    for (int i = 0; i < N; ++i) {
        int vid = key_idx[i];
        Vec3 pred = ed.deformVertexByState(key_old[i], x, B[vid], W[vid], 0);
        double e = (pred - key_new[i]).norm();
        sum += e; sum2 += e*e; mx = std::max(mx, e);
    }
    mean_e = sum / std::max(1, N);
    rmse_e = std::sqrt(sum2 / std::max(1, N));
    max_e  = mx;
}

// MATLAB-style sanity check: compare key_old to v_init(key_idx)
static void checkKeyOldVsVinit(const std::vector<Vec3>& v_init,
                               const std::vector<Vec3>& key_old,
                               const std::vector<int>& key_idx) {
    double sum=0.0, mx=0.0; int N = (int)key_old.size();
    for (int i=0;i<N;++i) {
        int vid = key_idx[i];
        if (vid < 0 || vid >= (int)v_init.size()) continue;
        double e = (key_old[i] - v_init[vid]).norm();
        sum += e; mx = std::max(mx, e);
    }
    double mean = (N>0)? (sum/N) : 0.0;
    std::cout << std::fixed << std::setprecision(6)
              << "[Check] key_old vs v_init(idx): mean=" << mean
              << "  max=" << mx << "\n";
}

static double smoothCost(const EDGraph& ed, const Eigen::VectorXd& x) {
    const int G = ed.numNodes();
    const auto& nodes = ed.getGraphNodes();
    const auto& neigh = ed.getNodeNeighbors();
    auto rowToMat3 = [](const double* row9){
        Eigen::Matrix3d A; A << row9[0],row9[1],row9[2], row9[3],row9[4],row9[5], row9[6],row9[7],row9[8]; return A; };

    double cost = 0.0;
    for (int i = 0; i < G; ++i) {
        const int bi = 12 * i;
        Eigen::Matrix3d Ai = rowToMat3(&x(bi));
        Vec3 ti(x(bi+9), x(bi+10), x(bi+11));
        const Vec3 gi = nodes[i].position;
        for (int j : neigh[i]) if (j > i) {
            const int bj = 12 * j;
            Vec3 tj(x(bj+9), x(bj+10), x(bj+11));
            const Vec3 gj = nodes[j].position;
            Vec3 lhs = Ai * (gj - gi) + gi + ti;
            Vec3 rhs = gj + tj;
            cost += 0.5 * (lhs - rhs).squaredNorm();
        }
    }
    return cost;
}

int main(int argc, char** argv) {
    // Pretty printing similar to MATLAB
    std::cout.setf(std::ios::fixed);

    // 0) Load original vertices (centered)
    std::vector<Vec3> v_init = loadXYZ("v_init.txt");
    if (v_init.empty()) {
        std::cerr << "[ERR] v_init.txt missing or empty. Export it from MATLAB after centering vout1." << std::endl;
        return -1;
    }

    // 1) Build MeshModel (preserves vertex order)
    MeshModel model;
    std::vector<MeshModel::Vertex> mesh_vs; mesh_vs.reserve(v_init.size());
    for (const auto& p : v_init) {
        MeshModel::Vertex v; v.x = p.x(); v.y = p.y(); v.z = p.z(); v.nx=v.ny=v.nz=0.0; mesh_vs.push_back(v);
    }
    model.setVertices(mesh_vs);

    // 2) Read K and nodes
    int K_bind = 6; loadInt("num_nearestpts.txt", K_bind);
    const int Ksmooth = std::max(1, K_bind - 1);

    EDGraph edgraph(/*K=*/K_bind);

    // Prefer MATLAB nodes.txt when available (exact node positions)
    std::vector<Vec3> matlab_nodes = loadXYZ("nodes.txt");
    if (!matlab_nodes.empty()) {
        std::vector<DeformationNode> nodes; nodes.reserve(matlab_nodes.size());
        for (const auto& p : matlab_nodes) {
            DeformationNode n; n.position = p; n.A.setIdentity(); n.t.setZero(); nodes.push_back(n);
        }
        edgraph.setGraphNodes(nodes);
        edgraph.bindVertices(model.getVertices());
        edgraph.buildKnnNeighbors(Ksmooth);
        std::cout << "[Nodes] Using MATLAB nodes.txt (" << nodes.size() << ")\n";
    } else {
        const double grid_size = 20.0; // default matches MATLAB
        edgraph.initializeGraph(model.getVertices(), grid_size);
        edgraph.buildKnnNeighbors(Ksmooth);
        std::cout << "[Nodes] MATLAB nodes.txt not found. Using voxel init." << "\n";
    }

    std::cout << "[Config] verts=" << model.getVertices().size()
              << "  nodes=" << edgraph.numNodes()
              << "  K_bind=" << K_bind
              << "  Ksmooth=" << Ksmooth << "\n";

    // 3) Load MATLAB keypoints and EXACT indices
    std::vector<Vec3> key_old = loadXYZ("key_old.txt");
    std::vector<Vec3> key_new = loadXYZ("key_new.txt");
    std::vector<int>  key_indices = loadIndices1Based("key_idx.txt");
    if (key_old.empty() || key_new.empty() || key_old.size() != key_new.size()) {
        std::cerr << "[ERR] key_old.txt/key_new.txt missing or size mismatch." << std::endl; return -1;
    }
    if (key_indices.size() != key_old.size()) {
        std::cerr << "[ERR] key_idx.txt size mismatch (expect same length as key_old)." << std::endl; return -1;
    }
    for (size_t i = 0; i < key_indices.size(); ++i) {
        if (key_indices[i] < 0 || key_indices[i] >= (int)mesh_vs.size()) {
            std::cerr << "[ERR] key_idx out of range at i=" << i << ": " << key_indices[i] << std::endl; return -1;
        }
    }

    // --- MATLAB-style sanity check line ---
    checkKeyOldVsVinit(v_init, key_old, key_indices);

    // 4) Initial state vector x (A=I, t=0 per node)
    const int G = edgraph.numNodes();
    Eigen::VectorXd x(12 * G);
    edgraph.writeToStateVector(x, 0);

    // 5) INITIAL metrics (match MATLAB precision)
    double kmean0, krmse0, kmax0; keyErrors(edgraph, key_old, key_new, key_indices, x, kmean0, krmse0, kmax0);
    double smooth0 = smoothCost(edgraph, x);
    std::cout << std::setprecision(6)
              << "[Init]   key_mean=" << kmean0
              << "  key_rmse=" << krmse0
              << "  key_max=" << kmax0
              << std::setprecision(8)
              << "  smooth_cost=" << smooth0 << "\n";

    // 6) Optimize — MATLAB-equivalent P-weighting (v_diag)
    OptimizerOptions opt; 
    opt.max_iters    = 80;
    // P weights exactly mirror MATLAB v_diag rows:
    opt.w_rot_rows   = 1.0;   // 6 rows (orthogonality)
    opt.w_conn_rows  = 0.1;   // connection rows (per neighbor edge)
    opt.w_data_rows  = 0.01;  // data rows (3 per key)
    // Line search params to mimic MATLAB LineSearch
    opt.alpha0 = 1.0;  // try full step first
    opt.step0  = 0.25; // adjust step size
    opt.gamma1 = 0.1;  // Armijo-like
    opt.gamma2 = 0.9;  // curvature-like

    Optimizer solver(opt);
    solver.optimize(edgraph, x, key_old, key_new, key_indices);

    // Save state vector for MATLAB inspection (row-major per node)
    saveVector("x_cpp.txt", x);
    std::cout << "Saved x_cpp.txt (" << x.size() << " values)\n";

    // 7) FINAL metrics
    double kmean1, krmse1, kmax1; keyErrors(edgraph, key_old, key_new, key_indices, x, kmean1, krmse1, kmax1);
    double smooth1 = smoothCost(edgraph, x);
    std::cout << std::setprecision(6)
              << "[Final]  key_mean=" << kmean1
              << "  key_rmse=" << krmse1
              << "  key_max=" << kmax1
              << std::setprecision(8)
              << "  smooth_cost=" << smooth1 << "\n";
    double drop = ((krmse0 - krmse1) / std::max(1e-12, krmse0) * 100.0);
    std::cout << std::setprecision(6)
              << "[Delta]  key_rmse_drop=" << drop << "%"
              << "  smooth_drop=" << (smooth0 - smooth1) << "\n";

    // 8) Save deformed points
    auto deformed = deformAll(edgraph, model.getVertices(), x);
    {
        std::ofstream fout("deformed_cpp.txt");
        for (const auto& p : deformed) fout << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
        std::cout << "Saved deformed_cpp.txt (" << deformed.size() << " points)\n";
    }

    return 0;
}