#include <jni.h>
#include <string>
#include <android/hardware_buffer_jni.h>
#include "vva/vva.h"

static vva::Engine* g_engine = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_vva_engine_VVAEngine_nativeInit(JNIEnv* env, jobject, jstring model_path) {
    if (!g_engine) g_engine = new vva::Engine();
    const char* path = env->GetStringUTFChars(model_path, nullptr);
    bool res = g_engine->loadModel(path);
    env->ReleaseStringUTFChars(model_path, path);
    return res;
}
