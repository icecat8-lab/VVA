void vva_put(int idx, float v) {
    float t;
    if (P_SLOPE > 0) {
        int ch = idx / P_PLANE;
        vec4 sv = vec4(wgt[P_SLOPE + (ch >> 2)]);
        int lane = ch & 3;
        float sl = lane == 0 ? sv.x : (lane == 1 ? sv.y : (lane == 2 ? sv.z : sv.w));
        t = (v > 0.0 ? v : v * sl) * P_SCALE;
    } else {
        t = vva_act(v, P_ALPHA) * P_SCALE;
    }
    if (P_HASRES != 0) {
        t += float(res[idx]);
    }
    dst[idx] = vva_store_t(t);
}

void vva_out(int base, int x, vva_vec4 lo, vva_vec4 hi) {
    if (x >= P_W) {
        return;
    }
    vva_put(base, float(lo.x));
    vva_put(base + P_PLANE, float(lo.y));
    vva_put(base + 2 * P_PLANE, float(lo.z));
    vva_put(base + 3 * P_PLANE, float(lo.w));
    vva_put(base + 4 * P_PLANE, float(hi.x));
    vva_put(base + 5 * P_PLANE, float(hi.y));
    vva_put(base + 6 * P_PLANE, float(hi.z));
    vva_put(base + 7 * P_PLANE, float(hi.w));
}

void vva_out4(int base, int x, vec4 v) {
    if (x >= P_W) {
        return;
    }
    vva_put(base, float(v.x));
    vva_put(base + P_PLANE, float(v.y));
    vva_put(base + 2 * P_PLANE, float(v.z));
    vva_put(base + 3 * P_PLANE, float(v.w));
}
