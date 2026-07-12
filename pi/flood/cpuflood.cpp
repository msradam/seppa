// CPU counterpart of vkflood2 for the concurrent-vs-sequential comparison:
// the same two-pass WCA2D update in float with OpenMP row parallelism, the
// same gates (double reference NMSE, mass vs rain, basin pooling), the same
// output format. Threads come from OMP_NUM_THREADS.
// Build: g++ -O3 -march=native -fopenmp -o cpuflood cpuflood.cpp
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

int main(int argc, char** argv){
    const uint32_t SZ = argc>1 ? atoi(argv[1]) : 256;
    const int N       = argc>2 ? atoi(argv[2]) : 400;
    const bool sim    = (argc>3 && std::string(argv[3])=="sim");
    const float k=0.20f, scale=0.01f;
    const float rain  = getenv("RAIN") ? atof(getenv("RAIN")) : 0.02f;
    const int threads = omp_get_max_threads();

    std::vector<float> H(SZ*SZ), w0(SZ*SZ,0.0f), w1(SZ*SZ,0.0f), fx(4*SZ*SZ,0.0f);
    float c=(SZ-1)*0.5f;
    for(uint32_t y=0;y<SZ;y++) for(uint32_t x=0;x<SZ;x++){uint32_t i=y*SZ+x; float dx=x-c,dy=y-c; H[i]=scale*(dx*dx+dy*dy);}

    float *a=w0.data(), *b=w1.data();
    auto t0=std::chrono::high_resolution_clock::now();
    for(int s=0;s<N;s++){
        #pragma omp parallel for schedule(static)
        for(uint32_t y=0;y<SZ;y++) for(uint32_t x=0;x<SZ;x++){
            uint32_t i=y*SZ+x; float hi=H[i]+a[i];
            float fL=(x>0)   ? std::max(0.0f,k*(hi-(H[i-1]+a[i-1])))   : 0;
            float fR=(x<SZ-1)? std::max(0.0f,k*(hi-(H[i+1]+a[i+1])))   : 0;
            float fU=(y>0)   ? std::max(0.0f,k*(hi-(H[i-SZ]+a[i-SZ]))) : 0;
            float fD=(y<SZ-1)? std::max(0.0f,k*(hi-(H[i+SZ]+a[i+SZ]))) : 0;
            float tot=fL+fR+fU+fD;
            float sc=a[i]/std::max(tot,std::max(a[i],1e-20f));
            fx[4*i]=fL*sc; fx[4*i+1]=fR*sc; fx[4*i+2]=fU*sc; fx[4*i+3]=fD*sc;
        }
        #pragma omp parallel for schedule(static)
        for(uint32_t y=0;y<SZ;y++) for(uint32_t x=0;x<SZ;x++){
            uint32_t i=y*SZ+x;
            float out=fx[4*i]+fx[4*i+1]+fx[4*i+2]+fx[4*i+3], in=0;
            if(x>0) in+=fx[4*(i-1)+1]; if(x<SZ-1) in+=fx[4*(i+1)];
            if(y>0) in+=fx[4*(i-SZ)+3]; if(y<SZ-1) in+=fx[4*(i+SZ)+2];
            b[i]=a[i]+rain-out+in;
        }
        std::swap(a,b);
    }
    double secs=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    double flops=22.0*(double)SZ*SZ*N;

    if(!sim){
        std::vector<double> da(SZ*SZ,0.0), db(SZ*SZ,0.0), dfx(4*SZ*SZ,0.0);
        for(int s=0;s<N;s++){
            for(uint32_t y=0;y<SZ;y++) for(uint32_t x=0;x<SZ;x++){uint32_t i=y*SZ+x; double hi=H[i]+da[i];
                double fL=(x>0)?std::max(0.0,(double)k*(hi-(H[i-1]+da[i-1]))):0, fR=(x<SZ-1)?std::max(0.0,(double)k*(hi-(H[i+1]+da[i+1]))):0;
                double fU=(y>0)?std::max(0.0,(double)k*(hi-(H[i-SZ]+da[i-SZ]))):0, fD=(y<SZ-1)?std::max(0.0,(double)k*(hi-(H[i+SZ]+da[i+SZ]))):0;
                double t=fL+fR+fU+fD; if(t>da[i]&&t>0){double sc=da[i]/t; fL*=sc;fR*=sc;fU*=sc;fD*=sc;} dfx[4*i]=fL;dfx[4*i+1]=fR;dfx[4*i+2]=fU;dfx[4*i+3]=fD;}
            for(uint32_t y=0;y<SZ;y++) for(uint32_t x=0;x<SZ;x++){uint32_t i=y*SZ+x; double out=dfx[4*i]+dfx[4*i+1]+dfx[4*i+2]+dfx[4*i+3], in=0;
                if(x>0) in+=dfx[4*(i-1)+1]; if(x<SZ-1) in+=dfx[4*(i+1)]; if(y>0) in+=dfx[4*(i-SZ)+3]; if(y<SZ-1) in+=dfx[4*(i+SZ)+2];
                db[i]=da[i]+rain-out+in;}
            std::swap(da,db);
        }
        double num=0,den=0,tot_f=0; double maxd=0; uint32_t argmax=0;
        for(uint32_t i=0;i<SZ*SZ;i++){double g=a[i]; double d=g-da[i]; num+=d*d; den+=da[i]*da[i]; tot_f+=g; if(g>maxd){maxd=g;argmax=i;}}
        double nmse=den>0?num/den:num, injected=(double)rain*N*SZ*SZ; uint32_t mx=argmax%SZ,my=argmax/SZ;
        printf("grid=%u^2  steps=%d  time=%.3fs  %.2f GFLOP/s (cpu, %d threads)\n", SZ,N,secs,flops/secs/1e9,threads);
        printf("correct(NMSE vs CPU)=%s  NMSE=%.2e\n", nmse<1e-3?"yes":"NO", nmse);
        printf("mass: rain_injected=%.1f  water_total=%.1f  (conserved vs rain=%s)\n",
               injected, tot_f, fabs(tot_f-injected)/injected<0.02?"yes":"NO");
        printf("pooling: max depth=%.3f at (%u,%u)  basin_centre=(%u,%u)  (pools in basin=%s)\n",
               maxd, mx,my, SZ/2,SZ/2, (abs((int)mx-(int)(SZ/2))<(int)SZ/8 && abs((int)my-(int)(SZ/2))<(int)SZ/8)?"yes":"no");
    } else {
        printf("grid=%u^2  steps=%d  time=%.3fs  %.2f GFLOP/s (cpu sim mode, %d threads)\n",
               SZ,N,secs,flops/secs/1e9,threads);
    }
    return 0;
}
