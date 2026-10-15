# VVA Framework (Visual Processing Architecture)

VVA is an AI inference engine purpose-built for **Super-Resolution and Image-to-Image tasks on Android**. 

The goal of this project is to eliminate the performance bottlenecks found in standard, general-purpose AI frameworks. VVA is designed to extract maximum performance from modern mobile hardware, focusing on extreme execution speed, memory efficiency, and flawless visual quality.

## Key Features

* **Zero-Copy Memory Pipeline:** Bypasses CPU-to-GPU memory copying by feeding images directly into the GPU/NPU via Android's `AHardwareBuffer`. This reduces device heating and prevents Out-Of-Memory (OOM) crashes when processing large, high-resolution images.
* **100% Quality Mixed Precision:** Automatically compresses primary model weights to FP16 for fast loading and execution, while intelligently retaining FP32 for precision-sensitive layers (such as LayerNorm or Output layers). This ensures perfect color accuracy and zero visual artifacts.
* **NPU & Vulkan First:** Offloads heavy tensor computations to the NPU via Android NNAPI, while utilizing Vulkan Compute Shaders for spatial upscaling and pixel manipulation.

---

## Project Structure

* `python/` - Toolkit for converting PyTorch models into the `.vva` format.
* `core/` - The C++ execution core (Engine, Memory Mapping, Shaders, Task Scheduler).
* `android/` - Android library module (.aar/.so) and sample project for seamless integration.

---

## 1. Model Conversion (Python CLI)

The conversion toolkit is lightweight, easy to install, and can run on a standard PC, Google Colab, or even mobile environments like Termux.

**Installation:**
Navigate to the Python directory and install the package:
```bash
cd vva-framework/python
pip install -e .
```

**Converting a Model:**
Prepare your PyTorch model (`.pth`) and run the following command:
```bash
vva-cli convert --model my_upscaler.pth --output my_upscaler.vva
```
The script will automatically analyze the layers, apply mixed precision rules, and export the model into the `.vva` format, ready for immediate mobile deployment.

---

## 2. Android Integration

Simply drop the `core/` and `android/vva-android/` directories into your Android Studio project.

**Project Setup (`settings.gradle`):**
```gradle
include ':app', ':vva-android'
```

**Usage (Java / Kotlin):**
Place your converted `.vva` file in the device storage or app assets, and pass the image data (using `HardwareBuffer`) directly to the engine.
```java
import com.vva.engine.VVAEngine;
import android.hardware.HardwareBuffer;

// 1. Initialize the engine and load the model
VVAEngine engine = new VVAEngine();
engine.init("/path/to/my_upscaler.vva");

// 2. Process the image via the Zero-Copy pipeline
// (inputBuffer and outputBuffer must be standard Android HardwareBuffers)
engine.processZeroCopy(inputBuffer, outputBuffer);
```

---

## Customization & Further Development

VVA is highly specialized for maximum speed, which means it doesn't support every possible AI operator out of the box. To deploy your specific upscaling model (e.g., Real-ESRGAN, SwinIR), you will need to configure two things in the C++ core:

1. **Add Custom Operators:** The core currently provides `Conv2D`, `PixelShuffle`, and `LeakyReLU`. If your model architecture uses different layers (like Attention Blocks or InstanceNorm), you must write the corresponding Vulkan Compute Shaders (`.comp`) or add the NNAPI mapping for those specific layers.
2. **Graph Execution Order:** You will need to update the logic in `vva-framework/core/src/graph.h` to link the operators and tensors together, matching the exact sequential flow of your specific model architecture.
