// JNI bridge for com.vva.engine.VVAEngine.
//
// Every native method takes the engine handle created by nativeCreate(). C++
// exceptions never cross the JNI boundary: each entry point catches everything
// and reports failure through the return value and nativeLastError().
#include <android/bitmap.h>
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <jni.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

#include "vva/vva.h"

namespace {

struct Handle {
    vva::Engine engine;
    std::string jniError;  // errors detected by this layer (bad bitmap format, size mismatch, ...)
};

inline Handle* toHandle(jlong h) { return reinterpret_cast<Handle*>(h); }

std::string asciiSafe(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (u < 0x20 || u >= 0x7F) c = '?';  // NewStringUTF needs valid modified UTF-8
    }
    return out;
}

// Calls ProgressListener.onProgress(float) -> boolean. A thrown exception cancels the job.
class ProgressBridge {
public:
    ProgressBridge(JNIEnv* env, jobject listener) : env_(env), listener_(listener) {
        if (!listener) return;
        jclass cls = env->GetObjectClass(listener);
        if (cls) {
            mid_ = env->GetMethodID(cls, "onProgress", "(F)Z");
            env->DeleteLocalRef(cls);
        }
        if (!mid_) env->ExceptionClear();
    }
    bool call(float fraction) {
        if (!mid_) return true;
        const jboolean keepGoing = env_->CallBooleanMethod(listener_, mid_, static_cast<jfloat>(fraction));
        if (env_->ExceptionCheck()) {
            env_->ExceptionClear();
            return false;
        }
        return keepGoing != JNI_FALSE;
    }

private:
    JNIEnv* env_;
    jobject listener_;
    jmethodID mid_ = nullptr;
};

// Installs the progress callback for the duration of one job.
class ScopedProgress {
public:
    ScopedProgress(vva::Engine& engine, ProgressBridge& bridge) : engine_(engine) {
        engine_.setProgressCallback([&bridge](float p) { return bridge.call(p); });
    }
    ~ScopedProgress() { engine_.setProgressCallback(nullptr); }

private:
    vva::Engine& engine_;
};

class BitmapLock {
public:
    BitmapLock(JNIEnv* env, jobject bitmap) : env_(env), bitmap_(bitmap) {
        ok_ = AndroidBitmap_lockPixels(env, bitmap, &pixels_) == ANDROID_BITMAP_RESULT_SUCCESS && pixels_ != nullptr;
    }
    ~BitmapLock() {
        if (ok_) AndroidBitmap_unlockPixels(env_, bitmap_);
    }
    bool ok() const { return ok_; }
    uint8_t* data() const { return static_cast<uint8_t*>(pixels_); }

private:
    JNIEnv* env_;
    jobject bitmap_;
    void* pixels_ = nullptr;
    bool ok_ = false;
};

class BufferLock {
public:
    BufferLock(AHardwareBuffer* buffer, uint64_t usage) : buffer_(buffer) {
        ok_ = AHardwareBuffer_lock(buffer, usage, -1, nullptr, &addr_) == 0 && addr_ != nullptr;
    }
    ~BufferLock() {
        if (ok_) AHardwareBuffer_unlock(buffer_, nullptr);
    }
    bool ok() const { return ok_; }
    uint8_t* data() const { return static_cast<uint8_t*>(addr_); }

private:
    AHardwareBuffer* buffer_;
    void* addr_ = nullptr;
    bool ok_ = false;
};

bool hasTranslucency(const uint8_t* p, int w, int h, size_t strideBytes) {
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = p + static_cast<size_t>(y) * strideBytes;
        for (int x = 0; x < w; ++x) {
            if (row[4 * x + 3] != 255) return true;
        }
    }
    return false;
}

void copyRows(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, int rows, size_t rowBytes) {
    for (int y = 0; y < rows; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * dstStride, src + static_cast<size_t>(y) * srcStride, rowBytes);
    }
}

void unpremultiply(uint8_t* px, size_t count) {
    for (size_t i = 0; i < count; ++i, px += 4) {
        const unsigned a = px[3];
        if (a == 255) continue;
        if (a == 0) {
            px[0] = px[1] = px[2] = 0;
            continue;
        }
        for (int c = 0; c < 3; ++c) {
            const unsigned v = (px[c] * 255u + a / 2u) / a;
            px[c] = static_cast<uint8_t>(v > 255u ? 255u : v);
        }
    }
}

void premultiply(uint8_t* px, size_t count) {
    for (size_t i = 0; i < count; ++i, px += 4) {
        const unsigned a = px[3];
        if (a == 255) continue;
        for (int c = 0; c < 3; ++c) px[c] = static_cast<uint8_t>((px[c] * a + 127u) / 255u);
    }
}

// Shared by the Bitmap and HardwareBuffer entry points. Rows are `inStride` / `outStride` bytes apart.
bool upscaleRgba(Handle* h, const uint8_t* inPixels, size_t inStride, int w, int ht, uint8_t* outPixels,
                 size_t outStride, bool inPremultiplied, bool outPremultiplied) {
    const int s = h->engine.scale();
    if (s <= 0) {
        h->jniError = "no model loaded";
        return false;
    }
    const size_t outW = static_cast<size_t>(w) * s;
    const int outH = ht * s;

    const bool translucent = hasTranslucency(inPixels, w, ht, inStride);
    const bool convertIn = translucent && inPremultiplied;
    const bool convertOut = translucent && outPremultiplied;

    std::vector<uint8_t> packedIn;
    const uint8_t* src = inPixels;
    if (inStride != static_cast<size_t>(w) * 4 || convertIn) {
        packedIn.resize(static_cast<size_t>(w) * ht * 4);
        copyRows(inPixels, inStride, packedIn.data(), static_cast<size_t>(w) * 4, ht, static_cast<size_t>(w) * 4);
        if (convertIn) unpremultiply(packedIn.data(), static_cast<size_t>(w) * ht);
        src = packedIn.data();
    }

    std::vector<uint8_t> packedOut;
    uint8_t* dst = outPixels;
    const bool tightOut = outStride == outW * 4 && !convertOut;
    if (!tightOut) {
        packedOut.resize(outW * outH * 4);
        dst = packedOut.data();
    }

    if (!h->engine.processImageTile(src, w, ht, dst, vva::DeviceType::AUTO)) return false;

    if (!tightOut) {
        if (convertOut) premultiply(packedOut.data(), outW * outH);
        copyRows(packedOut.data(), outW * 4, outPixels, outStride, outH, outW * 4);
    }
    return true;
}

// Classification of an RGBA image; unpremultiplies first when the bitmap carries premultiplied alpha.
bool classifyRgba(Handle* h, const uint8_t* pixels, size_t strideBytes, int w, int ht, bool premultiplied,
                  vva::Classification& result) {
    const bool convert = premultiplied && hasTranslucency(pixels, w, ht, strideBytes);
    std::vector<uint8_t> packed;
    const uint8_t* src = pixels;
    if (strideBytes != static_cast<size_t>(w) * 4 || convert) {
        packed.resize(static_cast<size_t>(w) * ht * 4);
        copyRows(pixels, strideBytes, packed.data(), static_cast<size_t>(w) * 4, ht, static_cast<size_t>(w) * 4);
        if (convert) unpremultiply(packed.data(), static_cast<size_t>(w) * ht);
        src = packed.data();
    }
    return h->engine.classify(src, w, ht, result);
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_vva_engine_VVAEngine_nativeCreate(JNIEnv*, jclass) {
    try {
        return reinterpret_cast<jlong>(new Handle());
    } catch (...) {
        return 0;
    }
}

JNIEXPORT void JNICALL Java_com_vva_engine_VVAEngine_nativeDestroy(JNIEnv*, jclass, jlong handle) {
    try {
        delete toHandle(handle);
    } catch (...) {
    }
}

JNIEXPORT jboolean JNICALL Java_com_vva_engine_VVAEngine_nativeInit(JNIEnv* env, jclass, jlong handle, jstring path) {
    Handle* h = toHandle(handle);
    if (!h) return JNI_FALSE;
    try {
        h->jniError.clear();
        if (!path) {
            h->jniError = "model path is null";
            return JNI_FALSE;
        }
        const char* chars = env->GetStringUTFChars(path, nullptr);
        if (!chars) return JNI_FALSE;  // OutOfMemoryError is pending
        const std::string p(chars);
        env->ReleaseStringUTFChars(path, chars);
        return h->engine.loadModel(p) ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        h->jniError = e.what();
    } catch (...) {
        h->jniError = "unknown native error";
    }
    return JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_vva_engine_VVAEngine_nativeLastError(JNIEnv* env, jclass, jlong handle) {
    Handle* h = toHandle(handle);
    if (!h) return nullptr;
    try {
        const std::string msg = !h->jniError.empty() ? h->jniError : h->engine.lastError();
        return env->NewStringUTF(asciiSafe(msg).c_str());
    } catch (...) {
        return nullptr;
    }
}

JNIEXPORT jboolean JNICALL Java_com_vva_engine_VVAEngine_nativeIsClassifier(JNIEnv*, jclass, jlong handle) {
    Handle* h = toHandle(handle);
    return (h && h->engine.isClassifier()) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL Java_com_vva_engine_VVAEngine_nativeClassLabel(JNIEnv* env, jclass, jlong handle, jint index) {
    Handle* h = toHandle(handle);
    if (!h) return nullptr;
    try {
        const std::vector<std::string> names = h->engine.classLabels();
        const std::string label = (index >= 0 && static_cast<size_t>(index) < names.size())
                                      ? names[static_cast<size_t>(index)]
                                      : "class " + std::to_string(index);
        return env->NewStringUTF(asciiSafe(label).c_str());
    } catch (...) {
        return nullptr;
    }
}

// Returns [index, confidence, probability_0, ..., probability_n-1], or null on failure.
JNIEXPORT jfloatArray JNICALL Java_com_vva_engine_VVAEngine_nativeClassifyBitmap(JNIEnv* env, jclass, jlong handle,
                                                                               jobject bitmap, jboolean premultiplied) {
    Handle* h = toHandle(handle);
    if (!h) return nullptr;
    try {
        h->jniError.clear();
        if (!bitmap) {
            h->jniError = "bitmap is null";
            return nullptr;
        }
        AndroidBitmapInfo info;
        std::memset(&info, 0, sizeof(info));
        if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
            h->jniError = "cannot read bitmap information";
            return nullptr;
        }
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            h->jniError = "the bitmap must be ARGB_8888 (software, not HARDWARE)";
            return nullptr;
        }
        BitmapLock lock(env, bitmap);
        if (!lock.ok()) {
            h->jniError = "cannot lock bitmap pixels";
            return nullptr;
        }
        vva::Classification result;
        if (!classifyRgba(h, lock.data(), info.stride, static_cast<int>(info.width), static_cast<int>(info.height),
                          premultiplied != JNI_FALSE, result)) {
            return nullptr;
        }
        std::vector<jfloat> values;
        values.push_back(static_cast<jfloat>(result.index));
        values.push_back(result.confidence);
        values.insert(values.end(), result.probabilities.begin(), result.probabilities.end());
        jfloatArray array = env->NewFloatArray(static_cast<jsize>(values.size()));
        if (!array) return nullptr;  // OutOfMemoryError is pending
        env->SetFloatArrayRegion(array, 0, static_cast<jsize>(values.size()), values.data());
        return array;
    } catch (const std::bad_alloc&) {
        h->jniError = "out of memory";
    } catch (const std::exception& e) {
        h->jniError = e.what();
    } catch (...) {
        h->jniError = "unknown native error";
    }
    return nullptr;
}

JNIEXPORT jint JNICALL Java_com_vva_engine_VVAEngine_nativeScale(JNIEnv*, jclass, jlong handle) {
    Handle* h = toHandle(handle);
    return h ? h->engine.scale() : 0;
}

JNIEXPORT void JNICALL Java_com_vva_engine_VVAEngine_nativeSetTileSize(JNIEnv*, jclass, jlong handle, jint v) {
    if (Handle* h = toHandle(handle)) h->engine.setTileSize(v);
}

JNIEXPORT void JNICALL Java_com_vva_engine_VVAEngine_nativeSetTilePad(JNIEnv*, jclass, jlong handle, jint v) {
    if (Handle* h = toHandle(handle)) h->engine.setTilePad(v);
}

JNIEXPORT void JNICALL Java_com_vva_engine_VVAEngine_nativeSetThreads(JNIEnv*, jclass, jlong handle, jint v) {
    if (Handle* h = toHandle(handle)) h->engine.setThreads(v);
}

JNIEXPORT void JNICALL Java_com_vva_engine_VVAEngine_nativeSetMemoryBudgetMb(JNIEnv*, jclass, jlong handle, jint v) {
    if (Handle* h = toHandle(handle)) h->engine.setMemoryBudgetMB(v);
}

JNIEXPORT void JNICALL Java_com_vva_engine_VVAEngine_nativeCancel(JNIEnv*, jclass, jlong handle) {
    if (Handle* h = toHandle(handle)) h->engine.cancel();
}

JNIEXPORT jboolean JNICALL Java_com_vva_engine_VVAEngine_nativeHasVulkan(JNIEnv*, jclass, jlong handle) {
    Handle* h = toHandle(handle);
    if (!h) return JNI_FALSE;
    try {
        return h->engine.hasVulkan() ? JNI_TRUE : JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

JNIEXPORT jstring JNICALL Java_com_vva_engine_VVAEngine_nativeVulkanDeviceName(JNIEnv* env, jclass, jlong handle) {
    Handle* h = toHandle(handle);
    if (!h) return nullptr;
    try {
        return env->NewStringUTF(asciiSafe(h->engine.vulkanDeviceName()).c_str());
    } catch (...) {
        return nullptr;
    }
}

JNIEXPORT jboolean JNICALL Java_com_vva_engine_VVAEngine_nativeProcessBitmap(JNIEnv* env, jclass, jlong handle,
                                                                           jobject inBitmap, jobject outBitmap,
                                                                           jobject listener, jboolean inPremul,
                                                                           jboolean outPremul) {
    Handle* h = toHandle(handle);
    if (!h) return JNI_FALSE;
    try {
        h->jniError.clear();
        const int s = h->engine.scale();
        if (s <= 0) {
            h->jniError = "no model loaded";
            return JNI_FALSE;
        }
        if (!inBitmap || !outBitmap) {
            h->jniError = "bitmap is null";
            return JNI_FALSE;
        }
        AndroidBitmapInfo ii;
        AndroidBitmapInfo oi;
        std::memset(&ii, 0, sizeof(ii));
        std::memset(&oi, 0, sizeof(oi));
        if (AndroidBitmap_getInfo(env, inBitmap, &ii) != ANDROID_BITMAP_RESULT_SUCCESS ||
            AndroidBitmap_getInfo(env, outBitmap, &oi) != ANDROID_BITMAP_RESULT_SUCCESS) {
            h->jniError = "cannot read bitmap information";
            return JNI_FALSE;
        }
        if (ii.format != ANDROID_BITMAP_FORMAT_RGBA_8888 || oi.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            h->jniError = "both bitmaps must be ARGB_8888 (software, not HARDWARE)";
            return JNI_FALSE;
        }
        if (ii.width == 0 || ii.height == 0 || oi.width != ii.width * static_cast<uint32_t>(s) ||
            oi.height != ii.height * static_cast<uint32_t>(s)) {
            h->jniError = "output bitmap must be exactly " + std::to_string(s) + "x the size of the input";
            return JNI_FALSE;
        }
        BitmapLock lockIn(env, inBitmap);
        BitmapLock lockOut(env, outBitmap);
        if (!lockIn.ok() || !lockOut.ok()) {
            h->jniError = "cannot lock bitmap pixels";
            return JNI_FALSE;
        }
        ProgressBridge bridge(env, listener);
        ScopedProgress scoped(h->engine, bridge);
        return upscaleRgba(h, lockIn.data(), ii.stride, static_cast<int>(ii.width), static_cast<int>(ii.height),
                           lockOut.data(), oi.stride, inPremul != JNI_FALSE, outPremul != JNI_FALSE)
                   ? JNI_TRUE
                   : JNI_FALSE;
    } catch (const std::bad_alloc&) {
        h->jniError = "out of memory";
    } catch (const std::exception& e) {
        h->jniError = e.what();
    } catch (...) {
        h->jniError = "unknown native error";
    }
    return JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_vva_engine_VVAEngine_nativeProcessBuffers(JNIEnv* env, jclass, jlong handle,
                                                                            jobject inBuffer, jobject outBuffer,
                                                                            jobject listener) {
    Handle* h = toHandle(handle);
    if (!h) return JNI_FALSE;
    try {
        h->jniError.clear();
        const int s = h->engine.scale();
        if (s <= 0) {
            h->jniError = "no model loaded";
            return JNI_FALSE;
        }
        AHardwareBuffer* ain = inBuffer ? AHardwareBuffer_fromHardwareBuffer(env, inBuffer) : nullptr;
        AHardwareBuffer* aout = outBuffer ? AHardwareBuffer_fromHardwareBuffer(env, outBuffer) : nullptr;
        if (!ain || !aout) {
            h->jniError = "HardwareBuffer is null";
            return JNI_FALSE;
        }
        AHardwareBuffer_Desc din;
        AHardwareBuffer_Desc dout;
        std::memset(&din, 0, sizeof(din));
        std::memset(&dout, 0, sizeof(dout));
        AHardwareBuffer_describe(ain, &din);
        AHardwareBuffer_describe(aout, &dout);
        if (din.format != AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM || dout.format != AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM ||
            din.layers != 1 || dout.layers != 1) {
            h->jniError = "both buffers must be single-layer RGBA_8888 HardwareBuffers";
            return JNI_FALSE;
        }
        if (din.width == 0 || din.height == 0 || dout.width != din.width * static_cast<uint32_t>(s) ||
            dout.height != din.height * static_cast<uint32_t>(s)) {
            h->jniError = "output buffer must be exactly " + std::to_string(s) + "x the size of the input";
            return JNI_FALSE;
        }
        BufferLock lockIn(ain, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN);
        if (!lockIn.ok()) {
            h->jniError = "cannot lock the input buffer for CPU reads (allocate it with USAGE_CPU_READ_OFTEN)";
            return JNI_FALSE;
        }
        BufferLock lockOut(aout, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN);
        if (!lockOut.ok()) {
            h->jniError = "cannot lock the output buffer for CPU writes (allocate it with USAGE_CPU_WRITE_OFTEN)";
            return JNI_FALSE;
        }
        ProgressBridge bridge(env, listener);
        ScopedProgress scoped(h->engine, bridge);
        // Hardware buffers carry straight (non-premultiplied) RGBA here; strides are in pixels.
        return upscaleRgba(h, lockIn.data(), static_cast<size_t>(din.stride) * 4, static_cast<int>(din.width),
                           static_cast<int>(din.height), lockOut.data(), static_cast<size_t>(dout.stride) * 4, false,
                           false)
                   ? JNI_TRUE
                   : JNI_FALSE;
    } catch (const std::bad_alloc&) {
        h->jniError = "out of memory";
    } catch (const std::exception& e) {
        h->jniError = e.what();
    } catch (...) {
        h->jniError = "unknown native error";
    }
    return JNI_FALSE;
}

}  // extern "C"
