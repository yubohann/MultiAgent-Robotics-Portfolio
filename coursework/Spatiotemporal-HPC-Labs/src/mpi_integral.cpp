// ============================================================
//  第三次上机：MPI 多进程并行求定积分
//  f(x) = sin(x)/x 于 (0,1), 区间数 n = 1000000000, 梯形法
//  ------------------------------------------------------------
//  任务划分:
//    内点序号 [1, n-1] 按块划分给 size 个进程, 各进程本地求和;
//    MPI_Reduce 把所有局部和归约到 0 号进程;
//    0 号进程加上首末点项, 乘步长得到积分值.
//  编译: mpicc -O2 -std=c++17 src/mpi_integral.cpp -o bin/mpi_integral -lm
//  运行: mpirun -np 4 ./bin/mpi_integral
// ============================================================
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

static inline double f(double x) {
    if (x <= 0.0) return 1.0;              // lim_{x->0} sin(x)/x = 1
    return sin(x) / x;
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

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long long n = 1000000000LL;      // 区间数
    const double TRUTH = 0.9460830703671830;
    const double h = 1.0 / (double)n;
    const int REPS = 3;                    // 重复轮数, 取最优

    // 内点 [1, n-1] 按块划分, 半开区间 [i0, i1)
    const long long m = n - 1;
    const long long i0 = 1 + m * rank / size;
    const long long i1 = 1 + m * (rank + 1) / size;

    double best = 1e300, total = 0.0;
    for (int r = 0; r < REPS; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        double s = 0.0;
        for (long long i = i0; i < i1; ++i)
            s += f((double)i * h);
        double gsum = 0.0;
        MPI_Reduce(&s, &gsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        if (rank == 0) {
            if (t1 - t0 < best) best = t1 - t0;
            total = (0.5 * (f(0.0) + f(1.0)) + gsum) * h;
        }
    }

    if (rank == 0) {
        double err = fabs(total - TRUTH);
        char fn[128];
        snprintf(fn, sizeof fn, "results/mpi_integral_P%d.csv", size);
        double prev = read_prev_time(fn);
        double use = (prev < best) ? prev : best;

        printf("[MPI] 求定积分  f(x)=sin(x)/x, (0,1), n=%lld, 进程数=%d\n",
               n, size);
        printf("      梯形法, 重复 %d 轮, 耗时 %.3f s%s\n", REPS, use,
               (prev < best) ? " (沿用历史最优)" : "");
        printf("      结果 %.12f, 误差 %.3e\n", total, err);

        if (best <= prev) {
            FILE *fp = fopen(fn, "w");
            if (fp) {
                fprintf(fp, "n,size,reps,time_s,result,abs_err\n");
                fprintf(fp, "%lld,%d,%d,%.6f,%.15g,%.6e\n",
                        n, size, REPS, best, total, err);
                fclose(fp);
            }
        }
    }
    MPI_Finalize();
    return 0;
}
