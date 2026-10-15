# VVA Framework (Visual Processing Architecture)

An AI inference engine built for **super-resolution and image-to-image tasks on Android**.

VVA targets the bottlenecks of general-purpose AI frameworks: execution speed, memory use, and visual fidelity when processing large images on mobile hardware.

## Key Features

- **Zero-copy memory pipeline** — images are passed straight to the GPU/NPU through Android's `AHardwareBuffer`, avoiding CPU-to-GPU copies. This lowers memory pressure and helps when processing large, high-resolution images.
- **Mixed precision (FP16 + FP32)** — main model weights are stored as FP16 for fast loading and execution, while precision-sensitive layers (such as LayerNorm or output layers) stay in FP32 to preserve color accuracy.
- **NPU + Vulkan** — heavy tensor computation is offloaded to the NPU through Android NNAPI, while Vulkan compute shaders handle spatial upscaling and pixel manipulation.

## How it works

```
PyTorch checkpoint (.pth)
        │  vva-convert
        ▼
my_upscaler.vva  +  my_upscaler.json
        │  VVAEngine.init()
        ▼
HardwareBuffer (input) ──► VVAEngine.processZeroCopy() ──► HardwareBuffer (output)
```

## Requirements

- **Converter:** Python 3 on a PC, Google Colab, or Termux.
- **Android:** the APIs VVA builds on set the minimum versions: `HardwareBuffer` needs Android 8.0 (API 26)+, NNAPI needs Android 8.1 (API 27)+, and Vulkan compute needs a Vulkan-capable GPU.

## Quick start

1. **Install** the converter (see [Installation](#1-installation)).
2. **Convert** your `.pth` checkpoint to `.vva` (see [Converting a model](#2-converting-a-model)).
3. **Add** the `core/` and `android/` modules to your Android project and call `VVAEngine` (see [Android integration](#3-android-integration)).

## Project structure

| Folder | Contents |
|---|---|
| `python/` | Toolkit for converting PyTorch models into the `.vva` format |
| `core/` | C++ execution core (engine, memory mapping, shaders, task scheduler) |
| `android/` | Android library module (`.aar` / `.so`) and sample project |

---

## 1. Installation

The conversion toolkit is lightweight and runs on a standard PC, Google Colab, or mobile environments such as Termux.

```bash
git clone https://github.com/icecat8-lab/VVA.git
cd VVA/python
pip install -e .
```

## 2. Converting a model

Prepare your PyTorch checkpoint (`.pth`) and run:

```bash
vva-convert --input /path/to/my_upscaler.pth --output /path/to/my_upscaler.vva
```

The tool inspects the layer topology and compiles the binary. It also writes a paired `.json` metadata file next to the `.vva` file for parsing in native code — keep both files together.

## 3. Android integration

### Add the module

In `settings.gradle`:

```gradle
include ':app', ':vva-android'
```

In your app module's `build.gradle`:

```gradle
dependencies {
    implementation project(':vva-android')
}
```

### Use it (Java)

Put the converted `.vva` file in your device storage or project assets, then initialize the engine and process images through the zero-copy pipeline:

```java
import com.vva.engine.VVAEngine;
import android.hardware.HardwareBuffer;

// 1. Initialize the engine and load the converted .vva model
VVAEngine engine = new VVAEngine();
engine.init("/path/to/my_upscaler.vva");

// 2. Process an image through the zero-copy pipeline
//    (inputBuffer and outputBuffer must both be Android HardwareBuffers)
engine.processZeroCopy(inputBuffer, outputBuffer);
```

### Use it (Kotlin)

```kotlin
import android.hardware.HardwareBuffer
import com.vva.engine.VVAEngine

val engine = VVAEngine()
engine.init("/path/to/my_upscaler.vva")
engine.processZeroCopy(inputBuffer, outputBuffer)
```

---

## Supported operators

| Operator | Status |
|---|---|
| `Conv2D` | Built in |
| `PixelShuffle` | Built in |
| `LeakyReLU` | Built in |
| Attention blocks, `InstanceNorm`, other layers | Need a custom implementation (see below) |

## Customization and further development

VVA is specialized for maximum execution speed, so it focuses on core operators out of the box. If your model architecture (such as Real-ESRGAN or SwinIR) uses specialized layers, configure them in the C++ core:

1. **Add custom operators.** For variants containing attention blocks or InstanceNorm, implement the corresponding Vulkan compute shaders (`.comp`) or NNAPI bindings.
2. **Update the graph execution flow.** Edit the sequential logic in `core/src/graph.h` to link operators and tensors according to your network topology.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
