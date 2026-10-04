// ============================================================
//  时空大数据高性能处理  ——  第一次上机：编程求定积分
//  ------------------------------------------------------------
//  被积函数 :  f(x) = sin(x)/x        (x=0 处按极限取 1)
//  积分区间 :  (a, b) = (0, 1)
//  理论真值 :  Si(1) = 0.9460830703671830...
//
//  对比算法 :
//    sumRect   矩形求和法(左端点)
//    sumTrap   梯形求和法
//    sumTrapK  Kahan 补偿求和的梯形法(附加对照)
//
//  对比维度 :
//    (1) 区间数 n = 1e3 / 1e6 / 1e9        -> 耗时与精度
//    (2) float vs double                    -> 舍入误差累积
//    (3) 编译优化 -O0 / -O2 / -O3+native    -> 性能差异
//
//  说明: 同时记录 墙钟时间(WALL) 与 进程CPU时间(CPU)。
//        在负载较高的机器上，CPU 时间更能反映算法本身的代价。
//
//  编译: g++ -O2 -std=c++17 -DUSE_FLOAT=0 integral_lab.cpp -o integral_double_O2
//  运行: ./integral_double_O2 --full
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <vector>
#include <string>

// ---------------- 精度开关 ----------------------------------
#ifndef USE_FLOAT
#define USE_FLOAT 1
#endif
#ifndef OPT_NAME
#define OPT_NAME "unknown"
#endif

#if USE_FLOAT
typedef float  Real;
#define REAL_NAME "float"
#else
typedef double Real;
#define REAL_NAME "double"
#endif

// ---------------- 被积函数 f(x) -----------------------------
static inline Real f(Real x)
{
    if (x <= (Real)0) return (Real)1;          // lim_{x->0} sin(x)/x = 1
    return (Real)(sin((double)x) / (double)x);
}

// ---------------- 矩形求和法(左端点) ------------------------
static Real sumRect(Real a, Real b, long long stepNum)
{
    const Real h = (Real)(((double)b - (double)a) / (double)stepNum);
    Real s = (Real)0;
    for (long long i = 0; i < stepNum; ++i)
        s += f(a + (Real)i * h);
    return s * h;
}

// ---------------- 梯形求和法 ---------------------------------
static Real sumTrap(Real a, Real b, long long stepNum)
{
    const Real h = (Real)(((double)b - (double)a) / (double)stepNum);
    Real s = (f(a) + f(b)) * (Real)0.5;
    for (long long i = 1; i < stepNum; ++i)
        s += f(a + (Real)i * h);
    return s * h;
}

// ---------------- 梯形 + Kahan 补偿求和(附加对照) -----------
static Real sumTrapKahan(Real a, Real b, long long stepNum)
{
    const Real h = (Real)(((double)b - (double)a) / (double)stepNum);
    Real s = (f(a) + f(b)) * (Real)0.5;
    Real c = (Real)0;                          // 补偿量
    for (long long i = 1; i < stepNum; ++i) {
        Real y = f(a + (Real)i * h) - c;
        Real t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s * h;
}

// ---------------- 高精度真值 Si(1) ---------------------------
static double trueSi(double x)
{
    double term = x, s = x, x2 = x * x;
    for (int k = 1; k < 500; ++k) {
        term *= -x2 / ((2.0 * k) * (2.0 * k + 1.0));
        double add = term / (2.0 * k + 1.0);
        s += add;
        if (fabs(add) < 1e-19) break;
    }
    return s;
}

// ---------------- 双时钟 -------------------------------------
static double wallSec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static double cpuSec()
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef Real (*IntegFunc)(Real, Real, long long);

int main(int argc, char** argv)
{
    const Real a = (Real)0, b = (Real)1;
    const double truth = trueSi(1.0);

    bool        full = false;
    long long   oneN = -1, repeat = -1;
    const char* sweepSpec = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--full"))                        full = true;
        else if (!strcmp(argv[i], "--n") && i + 1 < argc)      oneN   = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--list") && i + 1 < argc) {
            sweepSpec = argv[++i];
        }
    }

    std::vector<long long> steps;
    if (sweepSpec) {
        std::string s(sweepSpec);
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            std::string tok = (q == std::string::npos) ? s.substr(p) : s.substr(p, q - p);
            if (!tok.empty()) steps.push_back(atoll(tok.c_str()));
            if (q == std::string::npos) break;
            p = q + 1;
        }
    }
    else if (oneN > 0)   steps.push_back(oneN);
    else if (full)  steps = { 1000LL, 1000000LL, 1000000000LL };
    else            steps = { 1000LL, 1000000LL };

    struct Method { const char* name; IntegFunc fn; };
    Method methods[3] = {
        { "Rect",  sumRect       },
        { "Trap",  sumTrap       },
        { "TrapK", sumTrapKahan  }
    };

    printf("==============================================================\n");
    printf("  Integral of f(x)=sin(x)/x over (0,1)\n");
    printf("  precision : %-7s (%d bytes)   opt : %s\n",
           REAL_NAME, (int)sizeof(Real), OPT_NAME);
    printf("  exact     : %.16f\n", truth);
    printf("==============================================================\n\n");
    printf("%-6s %-12s %-5s %-11s %-11s %-9s %-18s %-11s\n",
           "method", "stepNum", "rep", "wall(s)", "cpu(s)", "minW", "result", "absErr");
    printf("---------------------------------------------------------------------------------------------\n");

    char csvPath[512];
    snprintf(csvPath, sizeof(csvPath), "results/raw_%s_%s.csv", REAL_NAME, OPT_NAME);
    FILE* fp = fopen(csvPath, "w");
    if (fp) fprintf(fp,
        "method,prec,opt,stepNum,repeat,wall_s,cpu_s,min_wall_s,result,absErr,relErr\n");

    for (size_t si = 0; si < steps.size(); ++si) {
        const long long n = steps[si];

        long long R = repeat;
        if (R <= 0) {
            if      (n <= 1000LL)      R = 50;
            else if (n <= 100000LL)    R = 30;
            else if (n <= 1000000LL)   R = 20;
            else if (n <= 10000000LL)  R = 10;
            else if (n <= 100000000LL) R = 5;
            else                       R = 3;
        }

        for (int m = 0; m < 3; ++m) {
            // 成对采样: 每轮记录 (wall, cpu)。以 CPU 时间为准挑选最优轮次，
            // 再报告该轮的 wall/cpu。这样两个数字来自同一次测量，避免
            // 各自独立取最小值导致 wall < cpu 的矛盾。
            std::vector<double> ws, cs;
            ws.reserve((size_t)R); cs.reserve((size_t)R);
            Real r = (Real)0;
            for (long long k = 0; k < R; ++k) {
                double w0 = wallSec(), c0 = cpuSec();
                r = methods[m].fn(a, b, n);
                double w1 = wallSec(), c1 = cpuSec();
                ws.push_back(w1 - w0);
                cs.push_back(c1 - c0);
            }
            size_t bi = 0;
            for (size_t k = 1; k < cs.size(); ++k)
                if (cs[k] < cs[bi]) bi = k;

            double wall = ws[bi], cpu = cs[bi];
            // 同时给出全部轮次的最小 wall（仅作参考）
            double minWall = *std::min_element(ws.begin(), ws.end());

            double res    = (double)r;
            double absErr = fabs(res - truth);
            double relErr = absErr / fabs(truth);

            printf("%-6s %-12lld %-5lld %-11.6f %-11.6f %-9.6f %-18.12f %-11.3e\n",
                   methods[m].name, n, R, wall, cpu, minWall, res, absErr);
            if (fp) fprintf(fp, "%s,%s,%s,%lld,%lld,%.9f,%.9f,%.9f,%.15g,%.6e,%.6e\n",
                            methods[m].name, REAL_NAME, OPT_NAME,
                            n, R, wall, cpu, minWall, res, absErr, relErr);
            fflush(stdout);
        }
        if (fp) fflush(fp);
    }
    if (fp) fclose(fp);

    printf("\n[CSV] %s\n", csvPath);
    return 0;
}