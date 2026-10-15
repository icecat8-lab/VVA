#include <vulkan/vulkan.h>
#include <vector>

namespace vva {

class VulkanDispatcher {
private:
    VkDevice device;
    VkCommandBuffer cmdBuffer;
    VkPipelineLayout pipelineLayout;
    VkPipeline computePipeline;

public:
    void dispatchCompute(uint32_t width, uint32_t height, uint32_t out_channels) {
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

        uint32_t groupX = (width + 15) / 16;
        uint32_t groupY = (height + 15) / 16;
        uint32_t groupZ = out_channels;

        vkCmdDispatch(cmdBuffer, groupX, groupY, groupZ);
    }

    void submitQueue(VkQueue queue) {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
    }
};

}
