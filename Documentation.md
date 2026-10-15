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

## 1. Installation & Model Conversion (CLI)

The conversion toolkit is lightweight, easy to install, and can run on a standard PC, Google Colab, or even mobile environments like Termux.

### Installation
Clone the repository and install the package locally:
```bash
git clone https://github.com/icecat8-lab/VVA.git
cd VVA/python
pip install -e .
```

### Converting a Model
Prepare your PyTorch model checkpoint (`.pth`) and execute the conversion command. The tool will automatically inspect the layer topology and compile the binary:
```bash
vva-convert --input /path/to/my_upscaler.pth --output /path/to/my_upscaler.vva
```
*Note: The script automatically generates a paired `.json` metadata file alongside your `.vva` binary for seamless parsing in native code.*

---

## 2. Android Integration Guide

To integrate the inference engine into your Android application, include the `core/` and `android/` modules within your Android Studio project structure.

### Project Setup (`settings.gradle`)
```gradle
include ':app', ':vva-android'
```

### Usage in Android (Java / Kotlin)
Place your compiled `.vva` model file into your device storage or project assets. Initialize the engine and process images using the Zero-Copy hardware buffer pipeline:

```java
import com.vva.engine.VVAEngine;
import android.hardware.HardwareBuffer;

// 1. Initialize the engine and load the converted .vva model
VVAEngine engine = new VVAEngine();
engine.init("/path/to/my_upscaler.vva");

// 2. Process image frames directly via the Zero-Copy pipeline
// (Both inputBuffer and outputBuffer must be standard Android AHardwareBuffers)
engine.processZeroCopy(inputBuffer, outputBuffer);
```

---

## Customization & Further Development

VVA is highly specialized for maximum execution speed, meaning it focuses on core operators out of the box. If your custom model architecture (such as Real-ESRGAN or SwinIR) utilizes specialized layers, configure them in the C++ core:

1. **Add Custom Operators:** The core natively supports `Conv2D`, `PixelShuffle`, and `LeakyReLU`. For architectural variants containing Attention Blocks or InstanceNorm, implement the corresponding Vulkan Compute Shaders (`.comp`) or NNAPI bindings.
2. **Graph Execution Flow:** Update the sequential logic inside `core/src/graph.h` to link operators and tensors according to your specific network topology.