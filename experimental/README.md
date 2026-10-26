# Experimental code (not part of any build)

These files were moved out of `core/` because they are unfinished and are **not**
compiled, tested or shipped. They are kept as notes for a future GPU / NPU backend.
Do not enable them as they are:

| File | Known problems |
|------|----------------|
| `nnapi_backend*.cpp` | `ANEURALNETWORKS_CONV_2D` is given 7 operands instead of 10 (padding left/right/top/bottom, strides and activation are missing); FP16 convolution needs NNAPI 1.2 (API 29), not API 27. |
| `vulkan_dispatcher.cpp` | Never creates a command buffer or a pipeline layout for the push constants it declares. |
| `vulkan_zero_copy.cpp` | Picks memory type 0 unconditionally, does not query `VkAndroidHardwareBufferFormatPropertiesANDROID`, and the device is created without `VK_ANDROID_external_memory_android_hardware_buffer`. |
| `shaders/*.comp` | Use fp16 storage/arithmetic that the device creation does not enable; the convolution accumulates in fp16 and clamps (replicates) at the border while the format specifies zero padding. |
| `scheduler.cpp`, `graph.h` | Empty skeletons. |

Any GPU backend has to be verified against the CPU executor with
`tests/test_engine.py` (the CPU executor is the reference) and on real hardware.
