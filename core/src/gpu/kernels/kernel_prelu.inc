int idx = int(gl_GlobalInvocationID.y) * P_GRIDW + int(gl_GlobalInvocationID.x);
if (idx >= P_COUNT) {
    return;
}
int ch = idx / P_PLANE;
vec4 sv = vec4(wgt[ch >> 2]);
int lane = ch & 3;
float sl = lane == 0 ? sv.x : (lane == 1 ? sv.y : (lane == 2 ? sv.z : sv.w));
float v = float(src[idx]);
dst[idx] = vva_store_t(v > 0.0 ? v : v * sl);
