#pragma once
#include <jni.h>
#include <android/hardware_buffer.h>
AHardwareBuffer* AHardwareBuffer_fromHardwareBuffer(JNIEnv*, jobject);
