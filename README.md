# VVA

A small super-resolution inference engine for Android. It runs `.vva` models
(a compact, checksummed model format with the graph included) on the phone's CPU,
tiles large images so memory stays bounded, and ships as an AAR for apps plus a
command-line tool.

* Runs the x4 **RRDBNet** of basicsr / Real-ESRGAN (and models trained with that code).
* **CPU only**, multi-threaded, float32 maths with fp16 weights on disk. No Vulkan or NNAPI
  is needed to process an image (Vulkan is only probed by `vva-cli --info`).
* Android 8.0+ (API 26), arm64-v8a. A 16 KB-page-aligned `libvva_engine.so`, no `libc++_shared.so`.
* Progress callback, cancellation, alpha channel, `Bitmap` and `HardwareBuffer` input/output.

> **Speed.** Every convolution runs on CPU cores, so heavy models (for example 23 RRDB
> blocks) are much slower than a GPU engine such as ncnn with Vulkan. Small and medium
> models are fine. Measure on your target phone; the memory budget and thread count are
> adjustable. A GPU backend is not included (see `experimental/README.md`).

## 1. Convert a model

```bash
pip install ./python            # numpy only
pip install torch               # only needed to read .pth files
vva-convert -i model.pth -o model.vva --name "my-model"
```

* Input: a `.pth` / `.pt` checkpoint (`params_ema`, `params` and `state_dict` wrappers and a
  `module.` prefix are handled) or a plain `.npz` of weights.
* `.pth` files are loaded with `weights_only=True` (PyTorch 1.13+); add `--allow-pickle` only for files you trust.
* After writing, the converter re-reads the file and checks the stored graph against the original
  network on a random image; it prints the numerical error and fails if they disagree.
* Options: `--fp32` (no fp16 rounding), `--lrelu-slope`, `--res-scale` (defaults 0.2, as in basicsr),
  `--no-verify`.
* Not supported (reported with a clear error): models with pixel-unshuffle (x2 / x1 variants),
  other architectures. The `.vva` is self-contained; there is no `.json` sidecar any more.

## 2. Use it in an app

Download `vva-android-<tag>.aar` from the GitHub release, put it in `app/libs/` and add:

```gradle
dependencies {
    implementation files('libs/vva-android-v1.0.0.aar')
}
```

Copy the model out of `assets/` once (the engine memory-maps a real file), then:

```java
File model = new File(getFilesDir(), "model.vva");
if (!model.exists()) {
    try (InputStream in = getAssets().open("model.vva");
         OutputStream out = new FileOutputStream(model)) {
        byte[] buf = new byte[1 << 16];
        for (int n; (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
    }
}

// On a background thread:
try (VVAEngine engine = new VVAEngine()) {
    if (!engine.init(model.getAbsolutePath())) {
        Log.e("VVA", engine.getLastError());
        return;
    }
    engine.setMemoryBudgetMb(256);                  // optional, default 320
    Bitmap big = engine.upscale(photo, fraction -> {
        runOnUiThread(() -> progressBar.setProgress((int) (fraction * 100)));
        return !userCancelled;                       // return false to cancel
    });
    if (big == null) Log.e("VVA", engine.getLastError());
}
```

Keep one `VVAEngine` for the lifetime of the screen instead of creating one per image.
`engine.cancel()` can be called from any thread. Other entry points:
`processBitmap(in, out, listener)` for your own output bitmap and
`processZeroCopy(HardwareBuffer in, HardwareBuffer out)` (CPU access to the buffers,
RGBA_8888, allocate with `USAGE_CPU_READ_OFTEN` / `USAGE_CPU_WRITE_OFTEN`).

Notes

* Call `init`/`upscale` from a worker thread; both block.
* Tile size is chosen automatically from the memory budget. For a 12 MP photo the image is
  processed in tiles and the result is written straight into the output bitmap.
  `setTilePad` (default 16) is the overlap used to hide tile borders.
* R8 / ProGuard rules are included in the AAR.

## 3. Command line (Termux, adb shell, Linux)

```bash
vva-cli info model.vva                                  # check a model
vva-cli run model.vva in.ppm out.ppm --threads 4        # upscale a P6 PPM image
vva-cli --info                                          # Vulkan diagnostics
```

PPM is used so the tool needs no image libraries; convert with `ffmpeg -i in.png in.ppm`.

## Building

GitHub Actions (`Build VVA`, manual trigger) runs all tests, builds the AAR, `libvva.so` and
`vva-cli` for arm64-v8a and attaches them to a release. Locally:

```bash
gradle :vva-android:assembleRelease                      # needs JDK 17, Android SDK + NDK 26.3
python3 tests/test_format.py && python3 tests/test_converter.py && \
python3 tests/test_engine.py && python3 tests/check_jni.py   # host tests (g++, numpy)
```

## What is tested

The host tests build the real C++ sources with AddressSanitizer + UBSan and compare against an
independent numpy implementation: RRDBNet outputs for odd image sizes and channel counts
(exercising every kernel remainder), every operator and convolution variant, fp16 conversion of
all special values, thread-count independence (bit-identical), tiling vs whole image, progress,
cancellation, alpha, memory-budget tiling, corrupted and hostile model files, the converter and
its error paths, and the JNI pixel handling (strides, premultiplied alpha). `check_jni.py` verifies
that the Java `native` declarations match the C++ functions.

Not testable off-device and therefore **not verified here**: the Android build itself
(NDK / Gradle / AAR packaging), JNI calls into a real `Bitmap` / `HardwareBuffer`, and speed on a
phone. Run the app once on a device before shipping.

## Layout

```
core/            model reader (model.cpp), CPU executor (exec.cpp), engine (vva.cpp)
android/         Gradle module + JNI (vva_jni.cpp) + Java API (VVAEngine.java)
cli/             vva-cli
python/          vva-convert, the .vva writer/reader, numpy reference implementation
docs/FORMAT.md   the .vva format
tests/           host tests
experimental/    unfinished GPU / NPU code (not built)
```

License: see `LICENSE`.
