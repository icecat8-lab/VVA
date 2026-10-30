// Epilogue helpers of the convolution kernel (global scope, after the push constants are declared).
void vva_put(int idx, float v) {
    float t = vva_act(v, P_ALPHA) * P_SCALE;
    if (P_HASRES != 0) {
        t += res[idx];
    }
    dst[idx] = t;
}

void vva_out(int base, int x, vec4 lo, vec4 hi) {
    if (x >= P_W) {
        return;
    }
    vva_put(base, lo.x);
    vva_put(base + P_PLANE, lo.y);
    vva_put(base + 2 * P_PLANE, lo.z);
    vva_put(base + 3 * P_PLANE, lo.w);
    vva_put(base + 4 * P_PLANE, hi.x);
    vva_put(base + 5 * P_PLANE, hi.y);
    vva_put(base + 6 * P_PLANE, hi.z);
    vva_put(base + 7 * P_PLANE, hi.w);
}
