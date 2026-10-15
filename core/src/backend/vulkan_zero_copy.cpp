#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>
#include <android/hardware_buffer.h>

namespace vva {

class VulkanZeroCopyManager {
private:
    VkDevice device;

public:
    VkImage bindAHardwareBufferToVulkan(AHardwareBuffer* buffer, VkDeviceMemory& outMemory) {
        AHardwareBuffer_Desc ahbDesc;
        AHardwareBuffer_describe(buffer, &ahbDesc);

        VkImportAndroidHardwareBufferInfoANDROID importInfo{};
        importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
        importInfo.buffer = buffer;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext = &importInfo;
        allocInfo.memoryTypeIndex = 0; 
        allocInfo.allocationSize = ahbDesc.stride * ahbDesc.height * 4;

        if (vkAllocateMemory(device, &allocInfo, nullptr, &outMemory) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }

        VkExternalMemoryImageCreateInfo extImageInfo{};
        extImageInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        extImageInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.pNext = &extImageInfo;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.extent = {ahbDesc.width, ahbDesc.height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        VkImage vkImage;
        vkCreateImage(device, &imageInfo, nullptr, &vkImage);
        vkBindImageMemory(device, vkImage, outMemory, 0);

        return vkImage;
    }
};

}
