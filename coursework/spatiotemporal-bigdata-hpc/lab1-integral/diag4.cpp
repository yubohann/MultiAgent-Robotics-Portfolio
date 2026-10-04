#include <cstdio>
#include <cmath>

static inline float f(float x){ if (x<=0.0f) return 1.0f; return (float)(sin((double)x)/(double)x); }

// 输出: n, s_final, exact_sum, result(=s*h), noop_count
int main(){
    long long ns[] = {
        1000LL, 5000LL, 10000LL, 50000LL, 100000LL, 200000LL, 500000LL,
        1000000LL, 1500000LL, 2000000LL, 3000000LL, 4000000LL, 5000000LL,
        7000000LL, 10000000LL, 12000000LL, 15000000LL, 16777216LL,
        20000000LL, 30000000LL, 50000000LL, 100000000LL, 500000000LL, 1000000000LL
    };
    printf("n,s_final,exact_sum,result,noop\n");
    for (long long n : ns){
        float h = (float)(1.0/(double)n);
        float s = 0.0f;
        double exact=0.0;
        long long noop=0;
        for (long long i=0;i<n;i++){
            float x = (float)((double)i*(double)h);
            float y = f(x);
            float before=s;
            s = s + y;
            if (s==before) noop++;
            exact += (double)y;
        }
        printf("%lld,%.4f,%.1f,%.9f,%lld\n", n,(double)s,exact,(double)(s*h),noop);
    }
    return 0;
}