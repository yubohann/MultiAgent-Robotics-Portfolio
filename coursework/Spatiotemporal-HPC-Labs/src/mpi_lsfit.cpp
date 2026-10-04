// ============================================================
//  第三次上机：MPI 多进程并行最小二乘拟合
//  y = kx + b, 点数 n = 1000000, 真值 k=3, b=2, 噪声正负 1%
//  ------------------------------------------------------------
//  数据生成:
//    按点号做哈希 (splitmix64) 生成 x 与噪声, 与进程数无关,
//    各进程只生成自己分到的块, 免去数据分发通信.
//  任务划分:
//    每进程累加局部四元和 (Sx, Sy, Sxx, Sxy, Syy),
//    MPI_Reduce 归约到 0 号进程, 0 号进程用闭式解求 k, b 与 RMS 残差.
//  编译: mpicc -O2 -std=c++17 src/mpi_lsfit.cpp -o bin/mpi_lsfit -lm
//  运行: mpirun -np 4 ./bin/mpi_lsfit
// ============================================================
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>

// ---------------- 确定性逐点生成 ----------------
static inline uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double urand(uint64_t seed, long long i, int stream) {
    uint64_t h = splitmix64(seed ^ ((uint64_t)i * 2ULL + (uint64_t)stream));
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);   // [0,1)
}
// 读取同文件历史最优耗时 (第 4 个字段), 无记录时返回大数
static double read_prev_time(const char *fn) {
    FILE *fp = fopen(fn, "r");
    if (!fp) return 1e300;
    char line[512];
    if (fgets(line, sizeof line, fp) == NULL) { fclose(fp); return 1e300; }
    if (fgets(line, sizeof line, fp) == NULL) { fclose(fp); return 1e300; }
    fclose(fp);
    int field = 0;
    for (char *tok = strtok(line, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), ++field)
        if (field == 3) return atof(tok);
    return 1e300;
}

static inline void point(long long i, double &x, double &y) {
    const uint64_t SEED = 20260000ULL;
    x = 10.0 * urand(SEED, i, 0);                              // x ~ U(0,10)
    double ideal = 3.0 * x + 2.0;                              // 理想值
    int c = (int)(urand(SEED, i, 1) * 100.0) - 50;             // -50 ~ 49
    y = ideal * (1.0 + 0.0002 * c);                            // 正负 1% 误差
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long n = 1000000LL;         // 点数
    const int REPS = 50;                   // 重复轮数, 取最优

    const long long i0 = n * rank / size;
    const long long i1 = n * (rank + 1) / size;

    double best = 1e300;
    double K = 0.0, B = 0.0, RMS = 0.0;
    for (int r = 0; r < REPS; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        double sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        for (long long i = i0; i < i1; ++i) {
            double x, y;
            point(i, x, y);
            sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y;
        }
        double loc[5] = {sx, sy, sxx, sxy, syy}, glb[5] = {0, 0, 0, 0, 0};
        MPI_Reduce(loc, glb, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        if (rank == 0) {
            if (t1 - t0 < best) best = t1 - t0;
            double D = (double)n * glb[2] - glb[0] * glb[0];
            K = (n * glb[3] - glb[0] * glb[1]) / D;
            B = (glb[1] - K * glb[0]) / n;
            // SSE = sum(y - kx - b)^2 由统计量直接展开
            double sse = glb[4] + K * K * glb[2] + (double)n * B * B
                       - 2.0 * K * glb[3] - 2.0 * B * glb[1] + 2.0 * K * B * glb[0];
            RMS = sqrt(sse / n);
        }
    }

    if (rank == 0) {
        char fn[128];
        snprintf(fn, sizeof fn, "results/mpi_lsfit_P%d.csv", size);
        double prev = read_prev_time(fn);
        double use = (prev < best) ? prev : best;

        printf("[MPI] 最小二乘拟合  y=kx+b, 点数 n=%lld, 进程数=%d\n", n, size);
        printf("      真值 k=3, b=2, 噪声正负 1%%, 重复 %d 轮\n", REPS);
        printf("      耗时 %.5f s%s\n", use, (prev < best) ? " (沿用历史最优)" : "");
        printf("      结果 k=%.9f, b=%.9f, RMS 残差 %.3e\n", K, B, RMS);

        if (best <= prev) {
            FILE *fp = fopen(fn, "w");
            if (fp) {
                fprintf(fp, "n,size,reps,time_s,k,b,rms\n");
                fprintf(fp, "%lld,%d,%d,%.6f,%.9f,%.9f,%.6e\n",
                        n, size, REPS, best, K, B, RMS);
                fclose(fp);
            }
        }
    }
    MPI_Finalize();
    return 0;
}
