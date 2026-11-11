int idx = int(gl_GlobalInvocationID.y) * P_GRIDW + int(gl_GlobalInvocationID.x);
if (idx >= P_COUNT) {
    return;
}
float v = vva_act(float(srcA[idx]), P_ALPHA) * P_SCALE;
if (P_HASB != 0) {
    v += float(srcB[idx]);
}
dst[idx] = vva_store_t(v);
