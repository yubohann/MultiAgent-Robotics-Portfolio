#include <cstdio>
#include <cmath>
static inline float f(float x){ if (x<=0.0f) return 1.0f; return (float)(sin((double)x)/(double)x); }
static double ulp_of(double s){ int e; frexpf((float)s,&e); return ldexp(1.0,e-24); }

int main(){
    // (a) 找 s 首次达到 2^22 的 n
    printf("=== (a) s 首次跨越 2^22=%.0f 的 n ===\n", pow(2,22));
    long long lo=4300000LL, hi=4600000LL;
    while(lo<hi){
        long long mid=(lo+hi)/2;
        float h=(float)(1.0/(double)mid); float s=0.0f; bool crossed=false;
        for(long long i=0;i<mid;i++){ s=s+f((float)((double)i*(double)h)); if((double)s>=pow(2,22)){crossed=true;break;} }
        if(crossed) hi=mid; else lo=mid+1;
    }
    printf("   最小 n (s>=2^22) = %lld\n", lo);
    {
        long long n=lo; float h=(float)(1.0/(double)n); float s=0.0f;
        for(long long i=0;i<n;i++) s=s+f((float)((double)i*(double)h));
        printf("   该处 s=%.1f  ulp=%.3f  s*h=%.9f\n",(double)s,ulp_of((double)s),(double)(s*h));
    }
    {
        long long n=lo-1; float h=(float)(1.0/(double)n); float s=0.0f;
        for(long long i=0;i<n;i++) s=s+f((float)((double)i*(double)h));
        printf("   前一档 n=%lld  s=%.1f  ulp=%.3f  s*h=%.9f\n",n,(double)s,ulp_of((double)s),(double)(s*h));
    }

    // (b) 结果 vs 1-delta/n 跨边界
    printf("\n=== (b) 跨越边界时的结果 ===\n");
    printf("%12s %16s %12s %14s %14s %12s\n","n","s_final","ulp@final","s*h","1-delta/n","误差");
    long long ns[]={4200000LL,4400000LL,4433000LL,4450000LL,4500000LL,4600000LL,4700000LL,4750000LL,5000000LL};
    for(long long n:ns){
        float h=(float)(1.0/(double)n); float s=0.0f;
        for(long long i=0;i<n;i++) s=s+f((float)((double)i*(double)h));
        double res=(double)(s*h); double d=(double)n-(double)s;
        printf("%12lld %16.1f %12.3f %14.9f %14.9f %12.3e\n",
               n,(double)s,ulp_of((double)s),res,1.0-d/n,fabs(res-0.9460830703671830));
    }

    // (c) 误差首次超过 1e-2 的 n
    printf("\n=== (c) float 误差首次超过 1e-2 ===\n");
    long long a=2000000LL,b=4000000LL;
    while(a<b){
        long long mid=(a+b)/2;
        float h=(float)(1.0/(double)mid); float s=0.0f;
        for(long long i=0;i<mid;i++) s=s+f((float)((double)i*(double)h));
        if(fabs((double)(s*h)-0.9460830703671830)>1e-2) b=mid; else a=mid+1;
    }
    printf("   最小 n (err>1e-2) = %lld\n", a);
    return 0;
}