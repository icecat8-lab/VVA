#pragma once
#include <jni.h>
#include <cstdint>
typedef struct { uint32_t width, height, stride; int32_t format; uint32_t flags; } AndroidBitmapInfo;
enum { ANDROID_BITMAP_FORMAT_RGBA_8888 = 1 };
enum { ANDROID_BITMAP_RESULT_SUCCESS = 0 };
int AndroidBitmap_getInfo(JNIEnv*, jobject, AndroidBitmapInfo*);
int AndroidBitmap_lockPixels(JNIEnv*, jobject, void**);
int AndroidBitmap_unlockPixels(JNIEnv*, jobject);
