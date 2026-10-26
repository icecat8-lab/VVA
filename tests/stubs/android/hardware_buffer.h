#pragma once
#include <cstdint>
struct AHardwareBuffer;
typedef struct { uint32_t width, height, layers, format; uint64_t usage; uint32_t stride, rfu0; uint64_t rfu1; } AHardwareBuffer_Desc;
enum { AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM = 1 };
enum : uint64_t { AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN = 3UL, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN = 3UL << 4 };
struct ARect;
void AHardwareBuffer_describe(const AHardwareBuffer*, AHardwareBuffer_Desc*);
int AHardwareBuffer_lock(AHardwareBuffer*, uint64_t, int32_t, const ARect*, void**);
int AHardwareBuffer_unlock(AHardwareBuffer*, int32_t*);
