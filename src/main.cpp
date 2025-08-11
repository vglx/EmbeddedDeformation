#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <random>
#include <algorithm>
#include <numeric>
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

static Eigen::MatrixXd toMat(const std::vector<Vec3>& pts) {
    Eigen::MatrixXd M(pts.size(), 3);
    for (size_t i = 0; i < pts.size(); ++i) M.row((int)i) = pts[i].transpose();
    return M;
}

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
    Eigen::Matrix3d Rtmp = V * U.transpose();
    if (Rtmp.determinant() < 0) { V.col(2) *= -1; Rtmp = V * U.transpose(); }
    R = Rtmp; t = ca - R * cb;
}

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
    std::vector<int> idx(X.rows()); std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(123); std::shuffle(idx.begin(), idx.end(), rng);
    Eigen::MatrixXd Y(maxN, X.cols());
    for (int i = 0; i < maxN; ++i) Y.row(i) = X.row(idx[i]);
    return Y;
}

static std::vector<Vec3> deformAll(const EDGraph& ed,
                                   const std::vector<MeshModel::Vertex>& V,
                                   const Eigen::VectorXd& x) {
    const auto& B = ed.getBindings();
    const auto& W = ed.getWeights();
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
    const auto& B = ed.getBindings();
    const auto& W = ed.getWeights();
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

static double smoothCost(const EDGraph& ed, const Eigen::VectorXd& x) {
    const int G = ed.numNodes();
    const auto& nodes = ed.getGraphNodes();
    const auto& neigh = ed.getNodeNeighbors();
    double cost = 0.0;
    for (int i = 0; i < G; ++i) {
        Eigen::Matrix<double,6,1> se3_i = x.segment<6>(6*i);
        Sophus::SE3d Ti = Sophus::SE3d::exp(se3_i);
        const Vec3 gi = nodes[i].position;
        for (int j : neigh[i]) if (j > i) {
            Eigen::Matrix<double,6,1> se3_j = x.segment<6>(6*j);
            Sophus::SE3d Tj = Sophus::SE3d::exp(se3_j);
            const Vec3 gj = nodes[j].position;
            Vec3 lhs = Ti.so3() * (gj - gi) + gi + Ti.translation();
            Vec3 rhs = gj + Tj.translation();
            cost += 0.5 * (lhs - rhs).squaredNorm();
        }
    }
    return cost;
}

int main(int argc, char** argv) {
    // --- 0) Read vertices directly from MATLAB export ---
    std::vector<Vec3> original_points = loadXYZ("v_init.txt");
    if (original_points.empty()) {
        std::cerr << "[ERR] v_init.txt missing or empty. Export it from MATLAB after centering vout1." << std::endl;
        return -1;
    }

    // 1) Build MeshModel with EXACT order
    MeshModel model;
    std::vector<MeshModel::Vertex> mesh_vs; mesh_vs.reserve(original_points.size());
    for (const auto& p : original_points) {
        MeshModel::Vertex v; v.x = p.x(); v.y = p.y(); v.z = p.z(); v.nx=v.ny=v.nz=0.0; mesh_vs.push_back(v);
    }
    model.setVertices(mesh_vs);

    // 2) Read K and nodes if present
    int K_bind = 6; loadInt("num_nearestpts.txt", K_bind);
    const int Ksmooth = std::max(1, K_bind - 1);

    EDGraph edgraph(/*K=*/K_bind);

    // Prefer MATLAB nodes.txt when available
    std::vector<Vec3> matlab_nodes = loadXYZ("nodes.txt");
    if (!matlab_nodes.empty()) {
        std::vector<DeformationNode> nodes; nodes.reserve(matlab_nodes.size());
        for (const auto& p : matlab_nodes) {
            DeformationNode n; n.position = p; n.transform = Sophus::SE3d(); nodes.push_back(n);
        }
        edgraph.setGraphNodes(nodes);            // exact MATLAB nodes
        edgraph.bindVertices(model.getVertices());
        edgraph.buildKnnNeighbors(Ksmooth);
        std::cout << "[Nodes] Using MATLAB nodes.txt (" << nodes.size() << ")\n";
    } else {
        // fallback: voxelize from vertices (kept for completeness)
        const double grid_size = 20.0; // default matches MATLAB
        edgraph.initializeGraph(model.getVertices(), grid_size);
        edgraph.buildKnnNeighbors(Ksmooth);
        std::cout << "[Nodes] MATLAB nodes.txt not found. Using voxel init.\n";
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

    // 4) Initial state
    const int G = edgraph.numNodes();
    Eigen::VectorXd x0(6 * G);
    edgraph.writeToStateVector(x0, 0);

    // 5) INITIAL metrics
    double kmean0, krmse0, kmax0; keyErrors(edgraph, key_old, key_new, key_indices, x0, kmean0, krmse0, kmax0);
    double smooth0 = smoothCost(edgraph, x0);
    std::cout << "[Init]   key_mean=" << kmean0 << "  key_rmse=" << krmse0 << "  key_max=" << kmax0
              << "  smooth_cost=" << smooth0 << "\n";

    // Optional: compare to MATLAB deformed, if available
    auto compareToMatlab = [&](const Eigen::VectorXd& x, const char* tag){
        std::ifstream test("deformed_matlab.txt"); if (!test.good()) return; // skip silently
        std::vector<Vec3> matlab_pts = loadXYZ("deformed_matlab.txt"); if (matlab_pts.empty()) return;
        auto def = deformAll(edgraph, model.getVertices(), x);
        Eigen::MatrixXd A = toMat(matlab_pts); Eigen::MatrixXd B = toMat(def);
        const int maxN = 120000; A = subsampleRows(A, maxN); B = subsampleRows(B, maxN);
        Eigen::Matrix3d R; Eigen::Vector3d t; kabschAlign(A, B, R, t);
        Eigen::MatrixXd B_aligned = (B * R.transpose()).rowwise() + t.transpose();
        Eigen::VectorXd dA = nnDists(A, B_aligned); Eigen::VectorXd dB = nnDists(B_aligned, A);
        double rmse = std::sqrt(dA.array().square().mean());
        double chamfer = dA.mean() + dB.mean();
        double haus = std::max(dA.maxCoeff(), dB.maxCoeff());
        std::cout << tag << "  RMSE=" << rmse << "  Chamfer=" << chamfer << "  Hausdorff=" << haus << "\n";
    };
    compareToMatlab(x0, "[Init-MATLAB]");

    // 6) Optimize (with MATLAB-style weights)
    Optimizer optimizer;
    Optimizer::Options opts; // initialize defaults
    opts.max_iters = 80;
    opts.lambda_init = 1e-4;
    opts.eps_jac = 1e-7;
    opts.tol_dx = 1e-6;
    // MATLAB Gauss-Newton uses diag weights: rotation:1, smooth:0.1, data:0.01 -> sqrt to residual scale
    opts.w_data            = 0.1;   // sqrt(0.01)
    opts.w_smooth          = 0.316; // sqrt(0.1)
    // opts.w_rot_ortho       = 1.0;   // keep small if needed (SE3 already orthogonal)
    // opts.w_rot_consistency = 0.316; // optional but useful
    opts.verbose = true;

    Eigen::VectorXd x_opt;
    optimizer.optimize(x0, x_opt, edgraph, key_old, key_new, key_indices, opts);

    // 7) FINAL metrics
    double kmean1, krmse1, kmax1; keyErrors(edgraph, key_old, key_new, key_indices, x_opt, kmean1, krmse1, kmax1);
    double smooth1 = smoothCost(edgraph, x_opt);
    std::cout << "[Final]  key_mean=" << kmean1 << "  key_rmse=" << krmse1 << "  key_max=" << kmax1
              << "  smooth_cost=" << smooth1 << "\n";
    std::cout << "[Delta]  key_rmse_drop=" << ((krmse0 - krmse1) / std::max(1e-12, krmse0) * 100.0) << "%"
              << "  smooth_drop=" << (smooth0 - smooth1) << "\n";

    compareToMatlab(x_opt, "[Final-MATLAB]");

    // 8) Save deformed
    auto deformed = deformAll(edgraph, model.getVertices(), x_opt);
    {
        std::ofstream fout("deformed_cpp.txt");
        for (const auto& p : deformed) fout << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
        std::cout << "Saved deformed_cpp.txt (" << deformed.size() << " points)\n";
    }

    // Optional PLY
    auto savePLY = [](const std::string& filename, const std::vector<Vec3>& points) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) { std::cerr << "Failed to open: " << filename << '\n'; return; }
        ofs << "ply\nformat ascii 1.0\n";
        ofs << "element vertex " << points.size() << "\n";
        ofs << "property float x\nproperty float y\nproperty float z\nend_header\n";
        for (const auto& p : points) ofs << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
        std::cout << "Saved PLY: " << filename << '\n';
    };
    savePLY("original.ply", original_points);
    savePLY("deformed.ply", deformed);

    return 0;
}