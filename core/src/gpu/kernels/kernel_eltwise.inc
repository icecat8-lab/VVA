// Element-wise: dst = act(a) * scale (+ b), over a whole tensor (borders stay zero because act(0) = 0).
int idx = int(gl_GlobalInvocationID.y) * P_GRIDW + int(gl_GlobalInvocationID.x);
if (idx >= P_COUNT) {
    return;
}
float v = vva_act(srcA[idx], P_ALPHA) * P_SCALE;
if (P_HASB != 0) {
    v += srcB[idx];
}
dst[idx] = v;
