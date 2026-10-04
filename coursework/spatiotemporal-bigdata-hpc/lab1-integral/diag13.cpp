#include <cstdio>
#include <cmath>
static inline float f(float x){ if (x<=0.0f) return 1.0f; return (float)(sin((double)x)/(double)x); }
int main(){
    const double T=0.9460830703671830;
    printf("=== float 误差扫描 (找首次超过 1e-2 的 n) ===\n");
    printf("%12s %16s %14s %10s\n","n","s*h","误差","超1e-2?");
    long long prev=-1;
    for(long long n=2000000LL;n<=3000000LL;n+=50000LL){
        float h=(float)(1.0/(double)n); float s=0.0f;
        for(long long i=0;i<n;i++) s=s+f((float)((double)i*(double)h));
        double e=fabs((double)(s*h)-T);
        printf("%12lld %16.9f %14.4e %10s\n",n,(double)(s*h),e,(e>1e-2?"YES":"no"));
    }
    return 0;
}