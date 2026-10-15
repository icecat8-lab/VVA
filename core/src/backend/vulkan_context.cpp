#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace vva {

class VulkanContext {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    std::string deviceName;

    bool ready() const { return device != VK_NULL_HANDLE; }

    bool init() {
        if (ready()) return true;

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "vva";
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            instance = VK_NULL_HANDLE;
            return false;
        }

        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            destroy();
            return false;
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // Pick the first device that has a compute-capable queue family.
        for (VkPhysicalDevice candidate : devices) {
            uint32_t familyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(familyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
            for (uint32_t i = 0; i < familyCount; ++i) {
                if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice = candidate;
                    queueFamilyIndex = i;
                    break;
                }
            }
            if (physicalDevice != VK_NULL_HANDLE) break;
        }
        if (physicalDevice == VK_NULL_HANDLE) {
            destroy();
            return false;
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        deviceName = props.deviceName;

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo{};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

        if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
            device = VK_NULL_HANDLE;
            destroy();
            return false;
        }
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);
        return true;
    }

    void destroy() {
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
        physicalDevice = VK_NULL_HANDLE;
        computeQueue = VK_NULL_HANDLE;
    }

    ~VulkanContext() { destroy(); }
};

}  // namespace vva
