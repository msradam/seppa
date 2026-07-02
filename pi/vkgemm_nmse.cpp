// Hardened vkgemm: RANDOM inputs + full double-precision CPU-reference NMSE.
// Replaces the original A=B=1 / two-element check (which a kernel that just
// stores SZ everywhere would pass) with a real numerical correctness gate that
// cannot be gamed by a partial or constant kernel. Correctness is checked at
// small sizes (256, 512) where the CPU reference is cheap; larger sizes are
// perf-only. NMSE tolerance 1e-3 (fp32 accumulation drift is ~1e-6). No deps
// beyond libvulkan. Build: g++ -O3 -o vkgemm_nmse vkgemm_nmse.cpp -lvulkan
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

#define VK_CHECK(x) do { VkResult r__=(x); if(r__!=VK_SUCCESS){fprintf(stderr,"VK %d @%d\n",r__,__LINE__);exit(1);} } while(0)

static std::vector<char> readFile(const char* p){ FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(1);} fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET); std::vector<char> b(s); fread(b.data(),1,s,f); fclose(f); return b; }

int main(int argc, char** argv){
    const char* spv = argc>1?argv[1]:"gemm.spv";
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
    VkInstance inst; VK_CHECK(vkCreateInstance(&ici,nullptr,&inst));
    uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,nullptr); std::vector<VkPhysicalDevice> pds(nd); vkEnumeratePhysicalDevices(inst,&nd,pds.data());
    VkPhysicalDevice phys=VK_NULL_HANDLE;
    for(auto pd:pds){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd,&p); if(std::string(p.deviceName).find("V3D")!=std::string::npos){phys=pd; printf("selected: %s\n",p.deviceName);} }
    if(!phys){fprintf(stderr,"no V3D\n");return 1;}
    uint32_t qf=UINT32_MAX,qn=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,nullptr); std::vector<VkQueueFamilyProperties> qfs(qn); vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,qfs.data());
    for(uint32_t i=0;i<qn;i++) if(qfs[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qf=i;break;}
    float pr=1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=qf; qci.queueCount=1; qci.pQueuePriorities=&pr;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
    VkDevice dev; VK_CHECK(vkCreateDevice(phys,&dci,nullptr,&dev)); VkQueue queue; vkGetDeviceQueue(dev,qf,0,&queue);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys,&mp);
    auto memtype=[&](uint32_t bits){ VkMemoryPropertyFlags w=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT; for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&w)==w) return i; exit(1); };
    auto code=readFile(spv);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=code.size(); smci.pCode=(const uint32_t*)code.data();
    VkShaderModule sh; VK_CHECK(vkCreateShaderModule(dev,&smci,nullptr,&sh));
    VkDescriptorSetLayoutBinding bnd[3]; for(int i=0;i<3;i++){bnd[i]={}; bnd[i].binding=i; bnd[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bnd[i].descriptorCount=1; bnd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dslci.bindingCount=3; dslci.pBindings=bnd;
    VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(dev,&dslci,nullptr,&dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,4};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount=1; plci.pSetLayouts=&dsl; plci.pushConstantRangeCount=1; plci.pPushConstantRanges=&pcr;
    VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(dev,&plci,nullptr,&pl));
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; ss.stage=VK_SHADER_STAGE_COMPUTE_BIT; ss.module=sh; ss.pName="main";
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.stage=ss; cpci.layout=pl;
    VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,nullptr,&pipe));
    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpc.queueFamilyIndex=qf; cpc.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool; VK_CHECK(vkCreateCommandPool(dev,&cpc,nullptr,&cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool=cpool; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
    VkCommandBuffer cmd; VK_CHECK(vkAllocateCommandBuffers(dev,&cbai,&cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; VK_CHECK(vkCreateFence(dev,&fci,nullptr,&fe));
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets=1; dpci.poolSizeCount=1; dpci.pPoolSizes=&dps; dpci.flags=VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    VkDescriptorPool dp; VK_CHECK(vkCreateDescriptorPool(dev,&dpci,nullptr,&dp));

    auto run_gemm=[&](uint32_t SZ, bool check){
        VkDeviceSize bytes=(VkDeviceSize)SZ*SZ*4;
        VkBuffer bufs[3]; VkDeviceMemory mems[3]; void* maps[3];
        for(int i=0;i<3;i++){
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size=bytes; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            VK_CHECK(vkCreateBuffer(dev,&bci,nullptr,&bufs[i]));
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,bufs[i],&mr);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size; mai.memoryTypeIndex=memtype(mr.memoryTypeBits);
            VK_CHECK(vkAllocateMemory(dev,&mai,nullptr,&mems[i])); VK_CHECK(vkBindBufferMemory(dev,bufs[i],mems[i],0));
            VK_CHECK(vkMapMemory(dev,mems[i],0,bytes,0,&maps[i]));
        }
        float* A=(float*)maps[0]; float* B=(float*)maps[1]; float* C=(float*)maps[2];
        // Seeded RANDOM inputs in [-1,1] -- a constant/partial kernel now fails NMSE.
        srand(SZ);
        for(uint32_t i=0;i<SZ*SZ;i++){A[i]=(float)rand()/RAND_MAX*2.0f-1.0f; B[i]=(float)rand()/RAND_MAX*2.0f-1.0f; C[i]=0.0f;}
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool=dp; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
        VkDescriptorSet ds; VK_CHECK(vkAllocateDescriptorSets(dev,&dsai,&ds));
        VkDescriptorBufferInfo dbi[3]={{bufs[0],0,bytes},{bufs[1],0,bytes},{bufs[2],0,bytes}}; VkWriteDescriptorSet w[3];
        for(int i=0;i<3;i++){w[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet=ds; w[i].dstBinding=i; w[i].descriptorCount=1; w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo=&dbi[i];}
        vkUpdateDescriptorSets(dev,3,w,0,nullptr);
        uint32_t G=SZ/32;
        auto rec=[&](int reps){
            VK_CHECK(vkResetCommandBuffer(cmd,0)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; VK_CHECK(vkBeginCommandBuffer(cmd,&bi));
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe); vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,nullptr);
            vkCmdPushConstants(cmd,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&SZ);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            for(int r=0;r<reps;r++){ vkCmdDispatch(cmd,G,G,1); if(r+1<reps) vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr); }
            VK_CHECK(vkEndCommandBuffer(cmd));
        };
        auto submit=[&](){ VK_CHECK(vkResetFences(dev,1,&fe)); VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
            auto t0=std::chrono::high_resolution_clock::now(); VK_CHECK(vkQueueSubmit(queue,1,&si,fe)); VK_CHECK(vkWaitForFences(dev,1,&fe,VK_TRUE,UINT64_MAX));
            return std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count(); };
        rec(1); double t1=submit();
        int reps=(int)(0.5/t1); if(reps<1)reps=1; if(reps>2000)reps=2000;
        rec(reps); double t=submit();
        double perIter=t/reps;
        double gflops=2.0*(double)SZ*SZ*SZ/perIter/1e9;
        // Full double-precision CPU reference NMSE at correctness sizes.
        double nmse=-1.0; const char* cs="perf";
        if(check){
            double num=0.0, den=0.0;
            for(uint32_t i=0;i<SZ;i++) for(uint32_t j=0;j<SZ;j++){
                double ref=0.0; for(uint32_t k=0;k<SZ;k++) ref+=(double)A[i*SZ+k]*(double)B[k*SZ+j];
                double d=(double)C[i*SZ+j]-ref; num+=d*d; den+=ref*ref;
            }
            nmse = den>0.0 ? num/den : num;
            cs = (nmse<1e-3) ? "yes" : "no";
        }
        printf("SZ=%-5u  %7.2f GFLOP/s  %6.1f ms/matmul  correct=%s NMSE=%.2e\n",
               SZ, gflops, perIter*1000.0, cs, nmse);
        for(int i=0;i<3;i++){ vkUnmapMemory(dev,mems[i]); vkDestroyBuffer(dev,bufs[i],nullptr); vkFreeMemory(dev,mems[i],nullptr); }
        vkFreeDescriptorSets(dev,dp,1,&ds);
    };
    printf("\nHardened SGEMM on V3D (random inputs, CPU-reference NMSE; ceiling 43.7 GFLOP/s):\n");
    run_gemm(256,true); run_gemm(512,true); run_gemm(1024,false);
    printf("\n(correct=yes requires NMSE < 1e-3 vs a double CPU reference on random data)\n");
    vkDestroyInstance(inst,nullptr);
    return 0;
}
