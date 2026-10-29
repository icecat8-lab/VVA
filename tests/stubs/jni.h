#pragma once
#include <cstdint>
typedef int32_t jint; typedef int64_t jlong; typedef uint8_t jboolean; typedef float jfloat;
typedef void* jobject; typedef void* jstring; typedef void* jclass; typedef void* jmethodID; typedef void* jfloatArray;
typedef int32_t jsize;
#define JNI_FALSE 0
#define JNI_TRUE 1
#define JNIEXPORT __attribute__((visibility("default")))
#define JNICALL
struct JNIEnv {
  const char* GetStringUTFChars(jstring, jboolean*);
  void ReleaseStringUTFChars(jstring, const char*);
  jstring NewStringUTF(const char*);
  jclass GetObjectClass(jobject);
  jmethodID GetMethodID(jclass, const char*, const char*);
  jboolean CallBooleanMethod(jobject, jmethodID, ...);
  jboolean ExceptionCheck();
  void ExceptionClear();
  void DeleteLocalRef(jobject);
  jfloatArray NewFloatArray(jsize);
  void SetFloatArrayRegion(jfloatArray, jsize, jsize, const jfloat*);
};
