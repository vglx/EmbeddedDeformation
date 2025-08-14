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

// MATLAB-style sanity check
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
    // 0) Load original vertices (centered)
    std::vector<Vec3> v_init = loadXYZ("v_init.txt");
    if (v_init.empty()) {
        std::cerr << "[ERR] v_init.txt missing or empty. Export it from MATLAB after centering vout1." << std::endl;
        return -1;
    }

    // 1) Build MeshModel
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

    // Prefer MATLAB nodes.txt when available
    std::vector<Vec3> matlab_nodes = loadXYZ("nodes.txt");
    if (!matlab_nodes.empty()) {
        std::vector<DeformationNode> nodes; nodes.reserve(matlab_nodes.size());
        for (const auto& p : matlab_nodes) {
            DeformationNode n; n.position = p; n.A.setIdentity(); n.t.setZero(); nodes.push_back(n);
        }
        edgraph.setGraphNodes(nodes);
        edgraph.bindVertices(model.getVertices());
        edgraph.buildKnnNeighbors(Ksmooth);
        // （可选）屏蔽多余行，贴近 MATLAB：不打印 [Nodes]/[Config]
    } else {
        const double grid_size = 20.0;
        edgraph.initializeGraph(model.getVertices(), grid_size);
        edgraph.buildKnnNeighbors(Ksmooth);
    }

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

    // 5) INITIAL metrics（保持与 MATLAB 小数位）
    double kmean0, krmse0, kmax0; keyErrors(edgraph, key_old, key_new, key_indices, x, kmean0, krmse0, kmax0);
    double smooth0 = smoothCost(edgraph, x);
    std::cout.setf(std::ios::fixed); // for this line only
    std::cout << std::setprecision(6)
              << "[Init]   key_mean=" << kmean0
              << "  key_rmse=" << krmse0
              << "  key_max=" << kmax0 << "\n";
    std::cout.unsetf(std::ios::floatfield);

    // 6) Optimize — MATLAB-equivalent P-weighting (v_diag)
    OptimizerOptions opt; 
    opt.max_iters    = 80;
    opt.w_rot_rows   = 1.0;
    opt.w_conn_rows  = 0.1;
    opt.w_data_rows  = 0.01;
    opt.alpha0 = 1.0;  // try full step first
    opt.step0  = 0.25; // adjust step size
    opt.gamma1 = 0.1;
    opt.gamma2 = 0.9;

    Optimizer solver(opt);
    solver.optimize(edgraph, x, key_old, key_new, key_indices);

    // Save state vector
    saveVector("x_cpp.txt", x);

    // 7) FINAL metrics（与 MATLAB 小数位靠近）
    double kmean1, krmse1, kmax1; keyErrors(edgraph, key_old, key_new, key_indices, x, kmean1, krmse1, kmax1);
    double smooth1 = smoothCost(edgraph, x);
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << "[Final]  key_mean=" << kmean1
              << "  key_rmse=" << krmse1
              << "  key_max=" << kmax1 << "\n";
    std::cout.unsetf(std::ios::floatfield);
    double drop = ((krmse0 - krmse1) / std::max(1e-12, krmse0) * 100.0);
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << "[Delta]  key_rmse_drop=" << drop << "%" << "\n";
    std::cout.unsetf(std::ios::floatfield);

    // 8) Save deformed points（可留，可去）
    auto deformed = deformAll(edgraph, model.getVertices(), x);
    {
        std::ofstream fout("deformed_cpp.txt");
        for (const auto& p : deformed) fout << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
    }

    return 0;
}