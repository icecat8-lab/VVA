// Tests the pixel plumbing of the JNI layer (row strides, premultiplied alpha) on the host.
// The JNI source is included directly; Android/JNI headers come from tests/stubs.
//   jni_logic_test <rrdbnet.vva>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "../android/vva-android/src/main/cpp/vva_jni.cpp"

// Definitions for the declarations in tests/stubs (never called in this test).
const char* JNIEnv::GetStringUTFChars(jstring, jboolean*) { return nullptr; }
void JNIEnv::ReleaseStringUTFChars(jstring, const char*) {}
jstring JNIEnv::NewStringUTF(const char*) { return nullptr; }
jclass JNIEnv::GetObjectClass(jobject) { return nullptr; }
jmethodID JNIEnv::GetMethodID(jclass, const char*, const char*) { return nullptr; }
jboolean JNIEnv::CallBooleanMethod(jobject, jmethodID, ...) { return 0; }
jboolean JNIEnv::ExceptionCheck() { return 0; }
void JNIEnv::ExceptionClear() {}
void JNIEnv::DeleteLocalRef(jobject) {}
jfloatArray JNIEnv::NewFloatArray(jsize) { return nullptr; }
void JNIEnv::SetFloatArrayRegion(jfloatArray, jsize, jsize, const jfloat*) {}
int AndroidBitmap_getInfo(JNIEnv*, jobject, AndroidBitmapInfo*) { return -1; }
int AndroidBitmap_lockPixels(JNIEnv*, jobject, void**) { return -1; }
int AndroidBitmap_unlockPixels(JNIEnv*, jobject) { return -1; }
void AHardwareBuffer_describe(const AHardwareBuffer*, AHardwareBuffer_Desc*) {}
int AHardwareBuffer_lock(AHardwareBuffer*, uint64_t, int32_t, const ARect*, void**) { return -1; }
int AHardwareBuffer_unlock(AHardwareBuffer*, int32_t*) { return -1; }
AHardwareBuffer* AHardwareBuffer_fromHardwareBuffer(JNIEnv*, jobject) { return nullptr; }

static int g_checks = 0, g_failed = 0;
#define CHECK(cond, msg)                                           \
    do {                                                           \
        ++g_checks;                                                \
        if (!(cond)) {                                             \
            ++g_failed;                                            \
            std::printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        }                                                          \
    } while (0)

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    Handle h;
    CHECK(h.engine.loadModel(argv[1]), "model loads");
    const int s = h.engine.scale();
    const int w = 7, ht = 5;
    const size_t inStride = w * 4 + 12, outW = static_cast<size_t>(w) * s, outStride = outW * 4 + 20;
    const int outH = ht * s;

    // Straight RGBA test image with some fully opaque, some translucent pixels.
    std::vector<uint8_t> straight(static_cast<size_t>(w) * ht * 4);
    unsigned seed = 7;
    for (size_t i = 0; i < straight.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        straight[i] = static_cast<uint8_t>(seed >> 24);
    }
    for (int i = 0; i < w * ht; ++i) straight[i * 4 + 3] = (i % 3 == 0) ? 255 : static_cast<uint8_t>(128 + (i * 37) % 127);

    auto strided = [&](const std::vector<uint8_t>& packed, size_t stride) {
        std::vector<uint8_t> out(stride * ht, 0xAB);
        for (int y = 0; y < ht; ++y) std::memcpy(&out[y * stride], &packed[static_cast<size_t>(y) * w * 4], w * 4);
        return out;
    };

    // Reference: the engine called directly on tightly packed straight data.
    std::vector<uint8_t> ref(outW * outH * 4);
    CHECK(h.engine.processImageTile(straight.data(), w, ht, ref.data(), vva::DeviceType::AUTO), "reference run");

    // 1. straight alpha, padded strides in and out: same pixels, padding bytes untouched.
    {
        std::vector<uint8_t> in = strided(straight, inStride);
        std::vector<uint8_t> out(outStride * outH, 0xCD);
        CHECK(upscaleRgba(&h, in.data(), inStride, w, ht, out.data(), outStride, false, false), "straight, padded strides");
        bool same = true, padOk = true;
        for (int y = 0; y < outH; ++y) {
            same &= std::memcmp(&out[y * outStride], &ref[y * outW * 4], outW * 4) == 0;
            for (size_t k = outW * 4; k < outStride; ++k) padOk &= out[y * outStride + k] == 0xCD;
        }
        CHECK(same, "strided output equals the packed engine output");
        CHECK(padOk, "row padding is not overwritten");
    }

    // 2. premultiplied in and out: invariant c <= alpha, alpha identical, opaque pixels unchanged.
    {
        std::vector<uint8_t> pre = straight;
        premultiply(pre.data(), static_cast<size_t>(w) * ht);
        std::vector<uint8_t> in = strided(pre, inStride);
        std::vector<uint8_t> out(outStride * outH, 0);
        CHECK(upscaleRgba(&h, in.data(), inStride, w, ht, out.data(), outStride, true, true), "premultiplied run");
        bool invariant = true, alphaSame = true, opaqueSame = true;
        for (int y = 0; y < outH; ++y) {
            for (size_t x = 0; x < outW; ++x) {
                const uint8_t* o = &out[y * outStride + x * 4];
                const uint8_t* r = &ref[(y * outW + x) * 4];
                invariant &= o[0] <= o[3] && o[1] <= o[3] && o[2] <= o[3];
                alphaSame &= o[3] == r[3];
                if (r[3] == 255) opaqueSame &= std::abs(o[0] - r[0]) <= 1 && std::abs(o[1] - r[1]) <= 1 && std::abs(o[2] - r[2]) <= 1;
            }
        }
        CHECK(invariant, "premultiplied output keeps colour <= alpha");
        CHECK(alphaSame, "alpha matches the engine's alpha upscale");
        CHECK(opaqueSame, "opaque pixels survive the premultiply round trip");
    }

    // 3. opaque image with premultiplied flags: nothing converted, alpha stays 255.
    {
        std::vector<uint8_t> opaque = straight;
        for (int i = 0; i < w * ht; ++i) opaque[i * 4 + 3] = 255;
        std::vector<uint8_t> out(outW * outH * 4, 0);
        CHECK(upscaleRgba(&h, opaque.data(), w * 4, w, ht, out.data(), outW * 4, true, true), "opaque premultiplied run");
        bool a = true;
        for (size_t i = 3; i < out.size(); i += 4) a &= out[i] == 255;
        CHECK(a, "opaque output stays opaque");
    }

    // 4. premultiply / unpremultiply round trip for reasonably opaque pixels.
    {
        bool ok = true;
        for (unsigned a = 128; a <= 255; ++a) {
            for (unsigned c = 0; c <= a; c += 5) {
                uint8_t px[4] = {static_cast<uint8_t>(c), static_cast<uint8_t>(c), static_cast<uint8_t>(c), static_cast<uint8_t>(a)};
                uint8_t orig[4] = {px[0], px[1], px[2], px[3]};
                unpremultiply(px, 1);
                premultiply(px, 1);
                ok &= std::abs(px[0] - orig[0]) <= 1;
            }
        }
        uint8_t zero[4] = {9, 9, 9, 0};
        unpremultiply(zero, 1);
        CHECK(ok, "premultiply round trip within one level");
        CHECK(zero[0] == 0 && zero[1] == 0 && zero[2] == 0, "alpha 0 clears colour");
    }

    // 5. engine errors surface and leave the handle usable.
    {
        Handle empty;
        std::vector<uint8_t> in(4, 0), out(16 * 4, 0);
        CHECK(!upscaleRgba(&empty, in.data(), 4, 1, 1, out.data(), 4, false, false) && empty.jniError == "no model loaded", "no model: clean failure");
    }

    // 6. classification through the JNI helper (row strides, premultiplied alpha).
    if (argc == 3) {
        Handle c;
        CHECK(c.engine.loadModel(argv[2]) && c.engine.isClassifier(), "classifier loads");
        const int cw = 23, ch = 17;
        std::vector<uint8_t> img(static_cast<size_t>(cw) * ch * 4);
        unsigned seed2 = 5;
        for (size_t i = 0; i < img.size(); ++i) {
            seed2 = seed2 * 1664525u + 1013904223u;
            img[i] = static_cast<uint8_t>(seed2 >> 24);
        }
        for (int i = 0; i < cw * ch; ++i) img[i * 4 + 3] = (i % 4 == 0) ? 255 : static_cast<uint8_t>(140 + (i * 13) % 100);

        vva::Classification direct;
        CHECK(c.engine.classify(img.data(), cw, ch, direct), "direct classify");

        const size_t stride = cw * 4 + 16;
        std::vector<uint8_t> padded(stride * ch, 0xAB);
        for (int y = 0; y < ch; ++y) std::memcpy(&padded[y * stride], &img[static_cast<size_t>(y) * cw * 4], cw * 4);
        vva::Classification viaStride;
        CHECK(classifyRgba(&c, padded.data(), stride, cw, ch, false, viaStride), "classify with padded rows");
        CHECK(viaStride.logits == direct.logits, "row padding does not change the result");

        std::vector<uint8_t> pre = img;
        premultiply(pre.data(), static_cast<size_t>(cw) * ch);
        vva::Classification viaPremul;
        CHECK(classifyRgba(&c, pre.data(), cw * 4, cw, ch, true, viaPremul), "classify premultiplied");
        float worst = 0.f;
        for (size_t i = 0; i < direct.logits.size(); ++i) worst = std::max(worst, std::fabs(direct.logits[i] - viaPremul.logits[i]));
        CHECK(worst < 0.5f * (std::fabs(direct.logits[0]) + std::fabs(direct.logits[1]) + 1.f), "premultiplied input gives nearly the same logits");

        Handle none;
        vva::Classification r;
        CHECK(!classifyRgba(&none, img.data(), cw * 4, cw, ch, false, r), "no model: clean failure");
    }

    std::printf("jni_logic_test: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
