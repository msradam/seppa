// Packed-state variant of vkflood: water and surface height live in one
// vec2 buffer (P = (W, H+W)), halving the flux pass's scattered scalar
// loads. Same two-pass WCA2D scheme, same gates (double CPU reference NMSE,
// mass conservation vs rain, basin pooling) as vkflood.cpp.
// Build: g++ -O3 -o vkflood2 vkflood2.cpp -lvulkan
// Shaders: flux2.comp / height2.comp (glslangValidator -V x.comp -o x.spv)
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>

#define VK_CHECK(x) do{VkResult r__=(x); if(r__!=VK_SUCCESS){fprintf(stderr,"VK %d @%d\n",r__,__LINE__);exit(1);} }while(0)
static std::vector<char> readFile(const char*p){FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open %s\n",p);exit(1);}fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);std::vector<char> b(s);fread(b.data(),1,s,f);fclose(f);return b;}
struct PC { uint32_t SZ; float k; float rain; };

static VkPipeline mkPipe(VkDevice dev, VkPipelineLayout pl, const char* spv){
    auto code=readFile(spv);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=code.size(); smci.pCode=(const uint32_t*)code.data();
    VkShaderModule sh; VK_CHECK(vkCreateShaderModule(dev,&smci,nullptr,&sh));
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=sh; ss.pName="main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.stage=ss; cpci.layout=pl;
    VkPipeline p; VK_CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,nullptr,&p)); return p;
}

int main(int argc, char** argv){
    const uint32_t SZ = argc>1 ? atoi(argv[1]) : 256;
    const int N       = argc>2 ? atoi(argv[2]) : 400;
    const bool sim    = (argc>3 && std::string(argv[3])=="sim");  // GPU-only: skip CPU verify
    const bool fused = getenv("FUSED") != nullptr;  // single fused.spv dispatch per step
    const char* fluxSpv   = getenv("FLUX_SPV")   ? getenv("FLUX_SPV")   : (fused ? "fused.spv" : "flux2.spv");
    const char* heightSpv = getenv("HEIGHT_SPV") ? getenv("HEIGHT_SPV") : "height2.spv";
    const float k=0.20f, scale=0.01f;
    const float rain  = getenv("RAIN") ? atof(getenv("RAIN")) : 0.02f;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    VkInstance inst; VK_CHECK(vkCreateInstance(&ici,nullptr,&inst));
    uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,nullptr); std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(inst,&nd,pds.data());
    VkPhysicalDevice phys=VK_NULL_HANDLE;
    for(auto pd:pds){VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd,&p); if(std::string(p.deviceName).find("V3D")!=std::string::npos){phys=pd; printf("selected: %s\n",p.deviceName);}}
    if(!phys){fprintf(stderr,"no V3D\n");return 1;}
    uint32_t qf=UINT32_MAX,qn=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,nullptr); std::vector<VkQueueFamilyProperties> qfs(qn); vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,qfs.data());
    for(uint32_t i=0;i<qn;i++) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qf=i;break;}
    float pr=1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=qf; qci.queueCount=1; qci.pQueuePriorities=&pr;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    VkDevice dev; VK_CHECK(vkCreateDevice(phys,&dci,nullptr,&dev)); VkQueue queue; vkGetDeviceQueue(dev,qf,0,&queue);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
    auto memtype=[&](uint32_t bits){VkMemoryPropertyFlags w=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT; for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&w)==w) return i; exit(1);};

    VkDescriptorSetLayoutBinding bnd[4]; for(int i=0;i<4;i++){bnd[i]={}; bnd[i].binding=i; bnd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bnd[i].descriptorCount=1; bnd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dslci.bindingCount=4; dslci.pBindings=bnd;
    VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(dev,&dslci,nullptr,&dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(PC)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(dev,&plci,nullptr,&pl));
    VkPipeline pFlux=mkPipe(dev,pl,fluxSpv), pHeight=fused?VK_NULL_HANDLE:mkPipe(dev,pl,heightSpv);

    // buffers: 0=H(float), 1=P0(vec2), 2=Flux(vec4), 3=P1(vec2)
    VkDeviceSize hbytes=(VkDeviceSize)SZ*SZ*4, pbytes=(VkDeviceSize)SZ*SZ*8, fbytes=(VkDeviceSize)SZ*SZ*16;
    VkBuffer buf[4]; VkDeviceMemory mem[4]; void* map[4];
    VkDeviceSize sizes[4]={hbytes,pbytes,fbytes,pbytes};
    for(int i=0;i<4;i++){
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size=sizes[i]; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VK_CHECK(vkCreateBuffer(dev,&bci,nullptr,&buf[i]));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf[i],&mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size; mai.memoryTypeIndex=memtype(mr.memoryTypeBits);
        VK_CHECK(vkAllocateMemory(dev,&mai,nullptr,&mem[i])); VK_CHECK(vkBindBufferMemory(dev,buf[i],mem[i],0));
        VK_CHECK(vkMapMemory(dev,mem[i],0,sizes[i],0,&map[i]));
    }
    float* H=(float*)map[0]; float* P0=(float*)map[1]; float* P1=(float*)map[3];
    const char* dempath=getenv("DEM");
    float la0=0,la1=0,lo0=0,lo1=0; bool geo=false;
    if(dempath){
        FILE* df=fopen(dempath,"r"); if(!df){fprintf(stderr,"DEM open %s\n",dempath);return 1;}
        int dn; if(fscanf(df,"%d %f %f %f %f",&dn,&la0,&la1,&lo0,&lo1)!=5){return 1;} geo=true;
        std::vector<float> dem(dn*dn); for(int j=0;j<dn*dn;j++) if(fscanf(df,"%f",&dem[j])!=1) return 1; fclose(df);
        float dmin=*std::min_element(dem.begin(),dem.end()), dmax=*std::max_element(dem.begin(),dem.end());
        for(uint32_t y=0;y<SZ;y++) for(uint32_t x=0;x<SZ;x++) H[y*SZ+x]=dem[(y*dn/SZ)*dn+(x*dn/SZ)]-dmin;
        printf("terrain: DEM %s %dx%d relief %.1f m\n",dempath,dn,dn,dmax-dmin);
    } else {
        float c=(SZ-1)*0.5f;
        for(uint32_t y=0;y<SZ;y++) for(uint32_t x=0;x<SZ;x++){uint32_t i=y*SZ+x; float dx=x-c,dy=y-c; H[i]=scale*(dx*dx+dy*dy);}
    }
    for(uint32_t i=0;i<SZ*SZ;i++){ P0[2*i]=0; P0[2*i+1]=H[i]; P1[2*i]=0; P1[2*i+1]=H[i]; }

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,8};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets=2; dpci.poolSizeCount=1; dpci.pPoolSizes=&dps;
    VkDescriptorPool dp; VK_CHECK(vkCreateDescriptorPool(dev,&dpci,nullptr,&dp));
    VkDescriptorSet ds[2]; VkDescriptorSetLayout dls[2]={dsl,dsl};
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool=dp; dsai.descriptorSetCount=2; dsai.pSetLayouts=dls;
    VK_CHECK(vkAllocateDescriptorSets(dev,&dsai,ds));
    // binding: 0=H, 1=PIn, 2=Flux, 3=POut
    auto wire=[&](VkDescriptorSet s,int pin,int pout){
        VkDescriptorBufferInfo b0{buf[0],0,hbytes}, b1{buf[pin],0,pbytes}, b2{buf[2],0,fbytes}, b3{buf[pout],0,pbytes};
        VkDescriptorBufferInfo bb[4]={b0,b1,b2,b3}; VkWriteDescriptorSet w[4];
        for(int i=0;i<4;i++){w[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet=s; w[i].dstBinding=i; w[i].descriptorCount=1; w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo=&bb[i];}
        vkUpdateDescriptorSets(dev,4,w,0,nullptr);
    };
    wire(ds[0],1,3); wire(ds[1],3,1);

    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpc.queueFamilyIndex=qf; VkCommandPool cpool; VK_CHECK(vkCreateCommandPool(dev,&cpc,nullptr,&cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool=cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cmd; VK_CHECK(vkAllocateCommandBuffers(dev,&cbai,&cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; VK_CHECK(vkCreateFence(dev,&fci,nullptr,&fe));

    const uint32_t strip = getenv("STRIP") ? atoi(getenv("STRIP")) : 1;  // cells per invocation in y
    uint32_t G=(SZ+15)/16, Gy=(SZ+16*strip-1)/(16*strip); PC pc{SZ,k,rain};
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; VK_CHECK(vkBeginCommandBuffer(cmd,&bi));
    for(int s=0;s<N;s++){
        VkDescriptorSet d=ds[s&1];
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pFlux);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&d,0,nullptr);
        vkCmdPushConstants(cmd,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(PC),&pc);
        vkCmdDispatch(cmd,G,Gy,1);
        if(!fused){
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pHeight);
            vkCmdDispatch(cmd,G,Gy,1);
        }
        if(s+1<N) vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
    }
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
    auto t0=std::chrono::high_resolution_clock::now();
    VK_CHECK(vkResetFences(dev,1,&fe)); VK_CHECK(vkQueueSubmit(queue,1,&si,fe)); VK_CHECK(vkWaitForFences(dev,1,&fe,VK_TRUE,UINT64_MAX));
    double secs=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    float* gpuP=(N&1)?P1:P0;

    if (!sim) {  // verification path (heavy CPU reference); "sim" mode skips it
    std::vector<double> a(SZ*SZ,0.0), b(SZ*SZ,0.0), fx(4*SZ*SZ,0.0);
    for(int s=0;s<N;s++){
        for(uint32_t y=0;y<SZ;y++) for(uint32_t x=0;x<SZ;x++){uint32_t i=y*SZ+x; double hi=H[i]+a[i];
            double fL=(x>0)?std::max(0.0,(double)k*(hi-(H[i-1]+a[i-1]))):0, fR=(x<SZ-1)?std::max(0.0,(double)k*(hi-(H[i+1]+a[i+1]))):0;
            double fU=(y>0)?std::max(0.0,(double)k*(hi-(H[i-SZ]+a[i-SZ]))):0, fD=(y<SZ-1)?std::max(0.0,(double)k*(hi-(H[i+SZ]+a[i+SZ]))):0;
            double t=fL+fR+fU+fD; if(t>a[i]&&t>0){double sc=a[i]/t; fL*=sc;fR*=sc;fU*=sc;fD*=sc;} fx[4*i]=fL;fx[4*i+1]=fR;fx[4*i+2]=fU;fx[4*i+3]=fD;}
        for(uint32_t y=0;y<SZ;y++) for(uint32_t x=0;x<SZ;x++){uint32_t i=y*SZ+x; double out=fx[4*i]+fx[4*i+1]+fx[4*i+2]+fx[4*i+3], in=0;
            if(x>0) in+=fx[4*(i-1)+1]; if(x<SZ-1) in+=fx[4*(i+1)]; if(y>0) in+=fx[4*(i-SZ)+3]; if(y<SZ-1) in+=fx[4*(i+SZ)+2];
            b[i]=a[i]+rain-out+in;}
        std::swap(a,b);
    }
    double num=0,den=0,tot_gpu=0,tot_cpu=0,maxd=0; uint32_t argmax=0;
    for(uint32_t i=0;i<SZ*SZ;i++){double g=gpuP[2*i]; double d=g-a[i]; num+=d*d; den+=a[i]*a[i]; tot_gpu+=g; tot_cpu+=a[i]; if(g>maxd){maxd=g;argmax=i;}}
    double nmse=den>0?num/den:num, injected=(double)rain*N*SZ*SZ; uint32_t mx=argmax%SZ,my=argmax/SZ;
    double flops=22.0*(double)SZ*SZ*N;
    printf("grid=%u^2  steps=%d  time=%.3fs  %.2f GFLOP/s\n", SZ,N,secs,flops/secs/1e9);
    printf("correct(NMSE vs CPU)=%s  NMSE=%.2e\n", nmse<1e-3?"yes":"NO", nmse);
    printf("mass: rain_injected=%.1f  water_total=%.1f  (conserved vs rain=%s)\n",
           injected, tot_gpu, fabs(tot_gpu-injected)/injected<0.02?"yes":"NO");
    printf("pooling: max depth=%.3f at (%u,%u)  basin_centre=(%u,%u)  (pools in basin=%s)\n",
           maxd, mx,my, SZ/2,SZ/2, (abs((int)mx-(int)(SZ/2))<SZ/8 && abs((int)my-(int)(SZ/2))<SZ/8)?"yes":"no");
    } else {
        printf("grid=%u^2  steps=%d  time=%.3fs  %.2f GFLOP/s (sim mode, no verify)\n",
               SZ,N,secs,22.0*(double)SZ*SZ*N/secs/1e9);
    }

    if(const char* fp=getenv("DUMP_FULL")){  // full-resolution depth grid (SZ*SZ float32), for visualization
        FILE* f=fopen(fp,"wb"); std::vector<float> w(SZ*SZ);
        for(uint32_t i=0;i<SZ*SZ;i++) w[i]=gpuP[2*i];
        fwrite(w.data(),4,SZ*SZ,f); fclose(f); printf("dumped %s (%ux%u float32)\n",fp,SZ,SZ);
    }
    { FILE* f=fopen("/tmp/flood_depth.txt","w"); const uint32_t D=16, step=SZ/D;
      fprintf(f,"# flood depth grid %ux%u, each cell = %ux%u sim cells, value = water depth (m)\n",D,D,step,step);
      if(geo) fprintf(f,"# bbox lat %.4f..%.4f lon %.4f..%.4f  row0=north col0=west\n",la0,la1,lo0,lo1);
      for(uint32_t gy=0;gy<D;gy++){ for(uint32_t gx=0;gx<D;gx++){
          uint32_t x=gx*step+step/2, y=gy*step+step/2; fprintf(f,"%5.2f ", gpuP[2*(y*SZ+x)]); } fprintf(f,"\n"); }
      fclose(f); printf("dumped /tmp/flood_depth.txt (16x16 depth grid)\n"); }
    vkDestroyInstance(inst,nullptr); return 0;
}
