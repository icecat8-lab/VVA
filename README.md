# VVA

VVA is a small inference engine for running image models on Android. It loads models in the
`.vva` format, a single checksummed file that contains both the network graph and its weights, and
runs them on the CPU. Large images are processed in tiles so memory use stays bounded.

The project provides an Android library (AAR), a command-line tool, and a converter for PyTorch
checkpoints.

## Limitations

* Processing runs on the CPU only. There is no GPU or NPU backend yet, so large models are slow:
  on a mid-range phone, the 6-block RRDBNet used for testing needs about 15 seconds to turn a
  128x192 image into 512x768 with 8 threads. Measure on your own target device.
* The Android build is arm64-v8a only and requires Android 8.0 (API 26) or newer.

## Converting a model

```bash
pip install ./python        # requires numpy
pip install torch           # only needed to read .pth files

vva-convert -i model.pth -o model.vva --name "my-model"
```

The input can be a `.pth` / `.pt` checkpoint or a plain `.npz` file of weights. Checkpoints are loaded
with `weights_only=True`; pass `--allow-pickle` only for files you trust. After writing the file, the
converter reads it back and compares the result with the original network, and it stops with an error
if the two disagree.

Useful options: `--fp32` stores weights at full precision (the default is fp16 and the file is half
the size), and `--no-verify` skips the check.

For a classifier, give the class names in output order and the preprocessing used in training:

```bash
vva-convert -i classifier.pth -o classifier.vva --classes photo,anime
```

The image is resized to `--input-size` (default 224) and normalised with `--mean` and `--std`
(default: ImageNet values). These settings are stored in the file and must match the training setup.

## Using the library in an app

Download `vva-android-<tag>.aar` from the Releases page, place it in `app/libs/`, and add it to
`build.gradle`:

```gradle
implementation files('libs/vva-android-<tag>.aar')
```

The engine reads the model from a real file, so copy it out of `assets/` once. Then, on a
background thread:

```java
try (VVAEngine engine = new VVAEngine()) {
    if (!engine.init(modelFile.getAbsolutePath())) {
        Log.e("VVA", engine.getLastError());
        return;
    }
    Bitmap result = engine.upscale(photo, fraction -> !cancelled);   // return false to cancel
    if (result == null) Log.e("VVA", engine.getLastError());
}
```

To choose a model automatically, load a classifier in a separate engine:

```java
VVAEngine.Classification c = classifier.classify(photo);
VVAEngine upscaler = (c != null && c.label.equals("anime")) ? animeEngine : photoEngine;
```

Notes:

* `init`, `upscale` and `classify` block, so never call them on the UI thread.
* Create one engine per model and keep it for the lifetime of the screen.
* `cancel()` may be called from any thread.
* `setMemoryBudgetMb()`, `setThreads()` and `setTileSize()` adjust memory use and speed.
* `processBitmap()` writes into a bitmap you supply, and `processZeroCopy()` works with
  `HardwareBuffer` objects (RGBA_8888, with CPU read/write usage flags).

## Command-line tool

`vva-cli` runs the same engine from Termux, `adb shell` or a Linux machine:

```bash
vva-cli info model.vva                              # check a model
vva-cli run model.vva in.ppm out.ppm --threads 8    # upscale an image
vva-cli classify classifier.vva in.ppm              # classify an image
```

Images use the PPM format so the tool needs no extra libraries. Convert with
`ffmpeg -i in.png in.ppm`.

## Building and testing

The GitHub Actions workflow `Build VVA` runs the tests, builds the AAR and the command-line tool,
and attaches them to a release. To build the library locally you need JDK 17, the Android SDK and
NDK 26.3:

```bash
gradle :vva-android:assembleRelease
```

The host tests need `g++`, `numpy` and `pillow`:

```bash
python3 tests/test_format.py
python3 tests/test_converter.py
python3 tests/test_engine.py
python3 tests/check_jni.py
```

They compile the real C++ sources with AddressSanitizer and UBSan and compare every operator, the
converter output and the classifier pipeline against independent numpy and PIL implementations.

Not covered by these tests: the Android build itself and calls into a real `Bitmap` or
`HardwareBuffer` on a device. Test the library inside an app before shipping it.

## License

See `LICENSE`.
