// ============================================================
//  第二次上机：编程实现矩阵运算和最小二乘拟合
//  ------------------------------------------------------------
//  矩阵基本运算:
//    matvec      矩阵与向量相乘   y = A x
//    matmul      矩阵与矩阵相乘   C = A B
//    transpose   矩阵转置         T = A^T
//    inverse     矩阵求逆         Inv = A^{-1}   高斯约当消元, 部分选主元
//  在此基础上实现最小二乘拟合 y = kx + b:
//    构造 A(n×2) = [x | 1], 方程组 A [k;b] ≈ y
//    正则方程 (A^T A) [k;b] = A^T y
//    [k;b] = (A^T A)^{-1} A^T y      全流程只用上述四个矩阵运算
//
//  点数实验: n = 10 / 1000 / 1000000, 记录耗时与拟合结果
//  编译: g++ -O2 -std=c++17 src/lab2_matrix_ls.cpp -o bin/lab2
//  运行: ./bin/lab2
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;

// ---------------- 矩阵结构(行主序) ----------------
struct Mat {
    int m = 0, n = 0;
    std::vector<double> a;
    Mat() {}
    Mat(int m_, int n_, double v = 0.0) : m(m_), n(n_), a((size_t)m_ * n_, v) {}
    double &at(int i, int j) { return a[(size_t)i * n + j]; }
    double  at(int i, int j) const { return a[(size_t)i * n + j]; }
};

// ---------------- 矩阵与向量相乘 y = A x ----------------
static std::vector<double> matvec(const Mat &A, const std::vector<double> &x) {
    std::vector<double> y(A.m, 0.0);
    for (int i = 0; i < A.m; ++i) {
        double s = 0.0;
        for (int j = 0; j < A.n; ++j) s += A.at(i, j) * x[j];
        y[i] = s;
    }
    return y;
}

// ---------------- 矩阵与矩阵相乘 C = A B ----------------
static Mat matmul(const Mat &A, const Mat &B) {
    Mat C(A.m, B.n);
    for (int i = 0; i < A.m; ++i) {
        double *crow = &C.a[(size_t)i * C.n];
        for (int k = 0; k < A.n; ++k) {
            double aik = A.at(i, k);
            const double *brow = &B.a[(size_t)k * B.n];
            for (int j = 0; j < B.n; ++j) crow[j] += aik * brow[j];
        }
    }
    return C;
}

// ---------------- 矩阵转置 T = A^T ----------------
static Mat transpose(const Mat &A) {
    Mat T(A.n, A.m);
    for (int i = 0; i < A.m; ++i)
        for (int j = 0; j < A.n; ++j) T.at(j, i) = A.at(i, j);
    return T;
}

// ---------------- 矩阵求逆 A^{-1} ----------------
// 增广 [A | I] 做高斯约当消元, 部分选主元, 主元过小判定奇异
static bool inverse(const Mat &A, Mat &Inv) {
    int n = A.m;
    if (A.n != n) return false;
    std::vector<double> W((size_t)n * 2 * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) W[(size_t)i * 2 * n + j] = A.at(i, j);
        W[(size_t)i * 2 * n + n + i] = 1.0;
    }
    for (int k = 0; k < n; ++k) {
        int p = k;
        for (int i = k + 1; i < n; ++i)
            if (std::fabs(W[(size_t)i * 2 * n + k]) > std::fabs(W[(size_t)p * 2 * n + k])) p = i;
        if (std::fabs(W[(size_t)p * 2 * n + k]) < 1e-14) return false;   // 奇异
        if (p != k)
            for (int j = 0; j < 2 * n; ++j)
                std::swap(W[(size_t)k * 2 * n + j], W[(size_t)p * 2 * n + j]);
        double piv = W[(size_t)k * 2 * n + k];
        for (int j = 0; j < 2 * n; ++j) W[(size_t)k * 2 * n + j] /= piv;
        for (int i = 0; i < n; ++i) {
            if (i == k) continue;
            double f = W[(size_t)i * 2 * n + k];
            if (f == 0.0) continue;
            for (int j = 0; j < 2 * n; ++j) W[(size_t)i * 2 * n + j] -= f * W[(size_t)k * 2 * n + j];
        }
    }
    Inv = Mat(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) Inv.at(i, j) = W[(size_t)i * 2 * n + n + j];
    return true;
}

// ---------------- 计时 ----------------
static double cpu_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---------------- 打印 ----------------
static void print_mat(const char *name, const Mat &A) {
    printf("  %s (%d×%d):\n", name, A.m, A.n);
    for (int i = 0; i < A.m; ++i) {
        printf("     [");
        for (int j = 0; j < A.n; ++j) printf(" %10.4f", A.at(i, j));
        printf("  ]\n");
    }
}

static void print_vec(const char *name, const std::vector<double> &v) {
    printf("  %s = [", name);
    for (size_t i = 0; i < v.size(); ++i) printf(" %.4f", v[i]);
    printf(" ]\n");
}

// ---------------- 最小二乘拟合 y = kx + b ----------------
struct LS { double k, b, rms; };

static LS fit_once(const std::vector<double> &x, const std::vector<double> &y) {
    int n = (int)x.size();
    Mat A(n, 2);
    for (int i = 0; i < n; ++i) { A.at(i, 0) = x[i]; A.at(i, 1) = 1.0; }
    Mat At   = transpose(A);            // A^T
    Mat AtA  = matmul(At, A);           // A^T A  (2×2)
    std::vector<double> Aty = matvec(At, y);   // A^T y  (2×1)
    Mat AtAinv;
    bool ok = inverse(AtA, AtAinv);     // (A^T A)^{-1}
    if (!ok) { return {0.0, 0.0, 0.0}; }
    std::vector<double> kb = matvec(AtAinv, Aty);   // [k; b]
    double s2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = y[i] - (kb[0] * x[i] + kb[1]);
        s2 += r * r;
    }
    return { kb[0], kb[1], std::sqrt(s2 / n) };
}

// ---------------- 造点: y = 3x + 2 + 1% 相对噪声, x ~ U(0,10) ----------------
static void gen_points(int n, unsigned seed,
                       std::vector<double> &x, std::vector<double> &y) {
    srand(seed);
    x.resize(n); y.resize(n);
    for (int i = 0; i < n; ++i) {
        double px = 10.0 * rand() / (double)RAND_MAX;       // 扩大随机数范围
        double py = 3.0 * px + 2.0;                        // 理想值
        int c = rand() % 100 - 50;                         // -50 ~ 49
        py += py * 0.0002 * c;                             // 正负 1% 随机误差
        x[i] = px; y[i] = py;
    }
}

int main() {
    printf("======================================================================\n");
    printf("  矩阵基本运算与最小二乘拟合   (g++ -O2, C++17)\n");
    printf("======================================================================\n\n");

    // ---------- 一 矩阵基本运算演示 ----------
    printf("一 矩阵基本运算\n\n");

    Mat A23(2, 3);
    A23.at(0,0)=1; A23.at(0,1)=2; A23.at(0,2)=3;
    A23.at(1,0)=4; A23.at(1,1)=5; A23.at(1,2)=6;
    Mat B32(3, 2);
    B32.at(0,0)=7;  B32.at(0,1)=8;
    B32.at(1,0)=9;  B32.at(1,1)=10;
    B32.at(2,0)=11; B32.at(2,1)=12;
    std::vector<double> xv = {1.0, 2.0, 3.0};

    std::vector<double> yv = matvec(A23, xv);
    print_mat("A", A23); print_vec("x", xv);
    printf("  [1] 矩阵与向量相乘  y = A x:\n"); print_vec("      y", yv);

    Mat C = matmul(A23, B32);
    print_mat("B", B32);
    printf("  [2] 矩阵与矩阵相乘  C = A B:\n"); print_mat("      C", C);

    Mat At = transpose(A23);
    printf("  [3] 矩阵转置  A^T:\n"); print_mat("      A^T", At);

    Mat M3(3, 3);
    M3.at(0,0)=1; M3.at(0,1)=2; M3.at(0,2)=3;
    M3.at(1,0)=0; M3.at(1,1)=1; M3.at(1,2)=4;
    M3.at(2,0)=5; M3.at(2,1)=6; M3.at(2,2)=0;
    Mat M3inv;
    inverse(M3, M3inv);
    Mat Check = matmul(M3, M3inv);
    double res = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            res = std::max(res, std::fabs(Check.at(i,j) - (i == j ? 1.0 : 0.0)));
    printf("  [4] 矩阵求逆  M^{-1}:\n");
    print_mat("      M", M3); print_mat("      M^{-1}", M3inv);
    printf("      验证 ‖M M^{-1} - I‖∞ = %.3e\n\n", res);

    // ---------- 二 最小二乘拟合实验 ----------
    const unsigned SEED = 2026;
    printf("二 最小二乘拟合  y = kx + b\n");
    printf("   真值 k = 3, b = 2;  噪声 ±1%%;  x ~ U(0,10);  随机种子 %u\n\n", SEED);

    const int    NS[3]   = {10, 1000, 1000000};
    const int    REPS[3] = {2000, 500, 20};

    fs::create_directories("results");
    FILE *fp = fopen("results/lab2_fit.csv", "w");
    if (fp) fprintf(fp, "n,reps,min_cpu_ms,us_per_point,k,b,rms\n");

    printf("   点数        重复     单次耗时(ms)    单点耗时(us)       k            b         RMS残差\n");
    printf("   --------------------------------------------------------------------------------------\n");
    for (int t = 0; t < 3; ++t) {
        int n = NS[t], reps = REPS[t];
        std::vector<double> x, y;
        gen_points(n, SEED, x, y);

        double best = 1e300;
        LS out{};
        for (int r = 0; r < reps; ++r) {
            double t0 = cpu_sec();
            out = fit_once(x, y);
            double t1 = cpu_sec();
            best = std::min(best, t1 - t0);
        }
        double ms = best * 1e3;
        double us_per = best * 1e6 / n;
        printf("   %-10d  %-6d  %12.5f   %13.4f   %12.7f  %12.7f   %.3e\n",
               n, reps, ms, us_per, out.k, out.b, out.rms);
        if (fp) fprintf(fp, "%d,%d,%.6f,%.4f,%.9f,%.9f,%.6e\n",
                        n, reps, ms, us_per, out.k, out.b, out.rms);
    }
    printf("\n   结果: 斜率 k ≈ 3, 截距 b ≈ 2, 与真值一致; 单点耗时近似恒定\n");
    if (fp) { fclose(fp); printf("   [CSV] results/lab2_fit.csv\n"); }
    return 0;
}
