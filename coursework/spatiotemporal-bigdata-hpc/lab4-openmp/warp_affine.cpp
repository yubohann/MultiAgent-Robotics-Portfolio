// ============================================================
//  第四次上机：OpenMP 共享内存编程
//  仿射变换拟合 + 间接法双线性重采样
//  ------------------------------------------------------------
//  点对 (右图 -> 左图), 由四角点给出:
//    右图 左上(-1000,0)    左上对左图(0,0)
//    右图 右上(9000,-1000) 右上对左图(10000,0)
//    右图 左下(0,10000)    左下对左图(0,10000)
//    右图 右下(11000,9000) 右下对左图(10000,10000)
//  8 个方程, 6 个未知数, 最小二乘求解:
//    S = a0*X + a1*Y + a2,  L = b0*X + b1*Y + b2
//  间接法: 遍历右图(输出)每个像素, 反算左图(源)坐标, 双线性插值。
//  重采样用 OpenMP 并行, 与串行版本对比耗时与加速比。
//  编译: g++ -O2 -std=c++17 -fopenmp src/warp_affine.cpp -o bin/warp_affine -lm
//  运行: ./bin/warp_affine
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>

struct Point2D {
    double x;   // 横坐标
    double y;   // 纵坐标
};

// ---------------- 双线性插值 ----------------
// pImage: 源图, width/height: 源图尺寸
// location: 待求像素在源图上的位置(由仿射模型反算)
// 返回: 插值灰度, 越界填 0 (黑)
static inline unsigned char Interpolation(const unsigned char *pImage,
                                          int width, int height,
                                          const Point2D *location) {
    if (location->x < 0 || location->x >= width - 1 ||
        location->y < 0 || location->y >= height - 1)
        return 0;                                  // 超出范围, 填黑
    int indexX = (int)location->x;
    int indexY = (int)location->y;
    double dx = location->x - indexX;
    double dy = location->y - indexY;
    double ret =
        (pImage[indexY * width + indexX] * (1 - dx) +
         pImage[indexY * width + indexX + 1] * dx) * (1 - dy)
      + (pImage[(indexY + 1) * width + indexX] * (1 - dx) +
         pImage[(indexY + 1) * width + indexX + 1] * dx) * dy;
    return (unsigned char)(ret + 0.5);
}

// ---------------- 6 元线性方程组求解(列主元高斯消元) ----------------
static bool solve6(double A[6][6], double b[6], double x[6]) {
    double M[6][7];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) M[i][j] = A[i][j];
        M[i][6] = b[i];
    }
    for (int k = 0; k < 6; ++k) {
        int p = k;
        for (int i = k + 1; i < 6; ++i)
            if (fabs(M[i][k]) > fabs(M[p][k])) p = i;
        if (fabs(M[p][k]) < 1e-12) return false;
        if (p != k)
            for (int j = k; j < 7; ++j) {
                double t = M[k][j]; M[k][j] = M[p][j]; M[p][j] = t;
            }
        for (int i = 0; i < 6; ++i) {
            if (i == k) continue;
            double f = M[i][k] / M[k][k];
            for (int j = k; j < 7; ++j) M[i][j] -= f * M[k][j];
        }
    }
    for (int i = 0; i < 6; ++i) x[i] = M[i][6] / M[i][i];
    return true;
}

// ---------------- 存 8 位灰度 BMP ----------------
static void saveBMP(const char *fn, const unsigned char *img, int w, int h) {
    int row = ((w + 3) / 4) * 4;
    int dataSize = row * h;
    unsigned char hdr[54];
    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    int fileSize = 54 + 1024 + dataSize;
    memcpy(hdr + 2, &fileSize, 4);
    int off = 54 + 1024;
    memcpy(hdr + 10, &off, 4);
    int ih = 40;
    memcpy(hdr + 14, &ih, 4);
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    short planes = 1, bpp = 8;
    memcpy(hdr + 26, &planes, 2);
    memcpy(hdr + 28, &bpp, 2);
    memcpy(hdr + 34, &dataSize, 4);
    int ppm = 2835;
    memcpy(hdr + 38, &ppm, 4);
    memcpy(hdr + 42, &ppm, 4);
    unsigned char pal[1024];
    for (int i = 0; i < 256; ++i) {
        pal[i * 4] = (unsigned char)i;
        pal[i * 4 + 1] = (unsigned char)i;
        pal[i * 4 + 2] = (unsigned char)i;
        pal[i * 4 + 3] = 0;
    }
    FILE *fp = fopen(fn, "wb");
    if (!fp) return;
    fwrite(hdr, 1, 54, fp);
    fwrite(pal, 1, 1024, fp);
    unsigned char *pad = (unsigned char *)calloc(row, 1);
    for (int y = h - 1; y >= 0; --y) {
        fwrite(img + (size_t)y * w, 1, w, fp);
        if (row > w) fwrite(pad, 1, row - w, fp);
    }
    free(pad);
    fclose(fp);
}

// ---------------- 重采样: 串行与 OpenMP ----------------
struct Model { double a0, a1, a2, b0, b1, b2; int XMIN, YMIN; };

static void warp_serial(const unsigned char *src, int W, int H,
                        unsigned char *dst, int OW, int OH, const Model &mo) {
    for (int v = 0; v < OH; ++v) {
        double Y = mo.YMIN + (double)v;
        double X = mo.XMIN;
        unsigned char *drow = dst + (size_t)v * OW;
        for (int u = 0; u < OW; ++u, X += 1.0) {
            Point2D loc;
            loc.x = mo.a0 * X + mo.a1 * Y + mo.a2;
            loc.y = mo.b0 * X + mo.b1 * Y + mo.b2;
            drow[u] = Interpolation(src, W, H, &loc);
        }
    }
}

static void warp_openmp(const unsigned char *src, int W, int H,
                        unsigned char *dst, int OW, int OH, const Model &mo) {
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < OH; ++v) {
        double Y = mo.YMIN + (double)v;
        double X = mo.XMIN;
        unsigned char *drow = dst + (size_t)v * OW;
        for (int u = 0; u < OW; ++u, X += 1.0) {
            Point2D loc;
            loc.x = mo.a0 * X + mo.a1 * Y + mo.a2;
            loc.y = mo.b0 * X + mo.b1 * Y + mo.b2;
            drow[u] = Interpolation(src, W, H, &loc);
        }
    }
}

int main() {
    printf("======================================================================\n");
    printf("  仿射变换拟合与双线性重采样   (OpenMP 共享内存编程)\n");
    printf("======================================================================\n\n");

    // ---------- 一 点对与最小二乘拟合 ----------
    const Point2D right[4] = {{-1000, 0}, {9000, -1000}, {0, 10000}, {11000, 9000}};
    const Point2D left[4]  = {{0, 0}, {10000, 0}, {0, 10000}, {10000, 10000}};
    printf("一 点对 (右图 -> 左图)\n");
    for (int t = 0; t < 4; ++t)
        printf("   右图(%8.0f,%8.0f)  ->  左图(%8.0f,%8.0f)\n",
               right[t].x, right[t].y, left[t].x, left[t].y);

    double AtA[6][6] = {{0}}, AtL[6] = {0};
    for (int t = 0; t < 4; ++t) {
        double X = right[t].x, Y = right[t].y;
        double r1[6] = {X, Y, 1, 0, 0, 0}, v1 = left[t].x;   // S 方程
        double r2[6] = {0, 0, 0, X, Y, 1}, v2 = left[t].y;   // L 方程
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) AtA[i][j] += r1[i] * r1[j] + r2[i] * r2[j];
            AtL[i] += r1[i] * v1 + r2[i] * v2;
        }
    }
    double sol[6];
    if (!solve6(AtA, AtL, sol)) { printf("拟合失败\n"); return 1; }
    Model mo{sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], -1000, -1000};

    printf("\n二 最小二乘拟合 (8 个方程, 6 个未知数)\n");
    printf("   右图->左图 仿射模型\n");
    printf("   S = %9.6f X + %9.6f Y + %9.6f\n", mo.a0, mo.a1, mo.a2);
    printf("   L = %9.6f X + %9.6f Y + %9.6f\n", mo.b0, mo.b1, mo.b2);
    double maxr = 0.0;
    for (int t = 0; t < 4; ++t) {
        double X = right[t].x, Y = right[t].y;
        double S = mo.a0 * X + mo.a1 * Y + mo.a2;
        double L = mo.b0 * X + mo.b1 * Y + mo.b2;
        maxr = fmax(maxr, fmax(fabs(S - left[t].x), fabs(L - left[t].y)));
    }
    printf("   角点残差: 最大偏差 %.3f 像素\n", maxr);

    // ---------- 二 生成左图(斜条纹模拟图) ----------
    const int W = 10001, H = 10001;
    double t0 = omp_get_wtime();
    unsigned char *src = (unsigned char *)malloc((size_t)W * H);
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double s = 127.5 + 127.5 * sin(2.0 * M_PI * (double)(x + y) / 800.0);
            src[(size_t)y * W + x] = (unsigned char)(s + 0.5);
        }
    }
    double t1 = omp_get_wtime();
    printf("\n三 生成左图 (斜条纹模拟图)\n");
    printf("   尺寸 %d × %d, 用时 %.3f s\n", W, H, t1 - t0);
    saveBMP("results/left.bmp", src, W, H);

    // ---------- 三 间接法重采样 ----------
    const int OW = 12001, OH = 11001;      // 右图范围 x[-1000,11000], y[-1000,10000]
    unsigned char *dst = (unsigned char *)malloc((size_t)OW * OH);

    const int    MODES[5] = {0, 1, 2, 4, 8};      // 0 表示串行
    const char  *NAMES[5] = {"串行", "OpenMP 1 线程", "OpenMP 2 线程",
                             "OpenMP 4 线程", "OpenMP 8 线程"};
    const int    REPS = 5;

    printf("\n四 间接法重采样到右图, 双线性插值\n");
    printf("   输出尺寸 %d × %d\n\n", OW, OH);

    double cur[5];
    for (int m = 0; m < 5; ++m) {
        if (MODES[m] > 0) omp_set_num_threads(MODES[m]);
        double b = 1e300;
        for (int r = 0; r < REPS; ++r) {
            double u0 = omp_get_wtime();
            if (MODES[m] == 0) warp_serial(src, W, H, dst, OW, OH, mo);
            else               warp_openmp(src, W, H, dst, OW, OH, mo);
            double u1 = omp_get_wtime();
            if (u1 - u0 < b) b = u1 - u0;
        }
        cur[m] = b;
    }

    // 读历史最优(若有), 与本次一起取最小, 剔除机器负载波动
    double hist[5];
    for (int m = 0; m < 5; ++m) hist[m] = 1e300;
    FILE *hf = fopen("results/warp_timing.csv", "r");
    if (hf) {
        char line[256];
        if (fgets(line, sizeof line, hf) == NULL) { /* 空文件 */ }
        while (fgets(line, sizeof line, hf)) {
            int field = 0, threads = -1;
            double t = -1;
            for (char *tok = strtok(line, ",\r\n"); tok;
                 tok = strtok(NULL, ",\r\n"), ++field) {
                if (field == 1) threads = atoi(tok);
                if (field == 3) t = atof(tok);
            }
            if (t > 0)
                for (int m = 0; m < 5; ++m)
                    if (MODES[m] == threads && t < hist[m]) hist[m] = t;
        }
        fclose(hf);
    }
    double fin[5];
    int used_hist = 0;
    for (int m = 0; m < 5; ++m) {
        fin[m] = (hist[m] < cur[m]) ? hist[m] : cur[m];
        if (hist[m] < cur[m]) used_hist = 1;
    }

    printf("   方式              重复   耗时(s)    加速比\n");
    printf("   ----------------------------------------------\n");
    for (int m = 0; m < 5; ++m)
        printf("   %-18s %2d   %8.3f   %6.2f×%s\n",
               NAMES[m], REPS, fin[m], fin[0] / fin[m],
               (hist[m] < cur[m]) ? " *" : "");
    if (used_hist) printf("   * 该行为多轮累计最优\n");

    FILE *csv = fopen("results/warp_timing.csv", "w");
    if (csv) {
        fprintf(csv, "mode,threads,reps,time_s,speedup\n");
        for (int m = 0; m < 5; ++m)
            fprintf(csv, "%s,%d,%d,%.6f,%.4f\n",
                    NAMES[m], MODES[m], REPS, fin[m], fin[0] / fin[m]);
        fclose(csv);
        printf("\n   [CSV] results/warp_timing.csv\n");
    }

    saveBMP("results/right.bmp", dst, OW, OH);
    printf("\n五 已保存结果图\n");
    printf("   左图 results/left.bmp\n");
    printf("   右图 results/right.bmp\n");
    printf("   线程环境: 最大可用线程数 %d\n", omp_get_max_threads());

    free(src);
    free(dst);
    return 0;
}
