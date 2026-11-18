int x0 = int(gl_GlobalInvocationID.x) * 4;
int y = int(gl_GlobalInvocationID.y) + P_YBASE;
int ocb = int(gl_GlobalInvocationID.z);
if (x0 >= P_W || y >= P_H) {
    return;
}
int pw = P_W + 2;
vva_vec4 a0l = vva_vec4(0.0);
vva_vec4 a0h = vva_vec4(0.0);
vva_vec4 a1l = vva_vec4(0.0);
vva_vec4 a1h = vva_vec4(0.0);
vva_vec4 a2l = vva_vec4(0.0);
vva_vec4 a2h = vva_vec4(0.0);
vva_vec4 a3l = vva_vec4(0.0);
vva_vec4 a3h = vva_vec4(0.0);
for (int ic = 0; ic < P_CIN; ++ic) {
    int wBase = (ocb * P_CIN + ic) * 18;
    int rBase = ic * P_PLANE + y * pw + x0;
    for (int ky = 0; ky < 3; ++ky) {
        int r = rBase + ky * pw;
        vva_real v0 = vva_real(src[r]);
        vva_real v1 = vva_real(src[r + 1]);
        vva_real v2 = vva_real(src[r + 2]);
        vva_real v3 = vva_real(src[r + 3]);
        vva_real v4 = vva_real(src[r + 4]);
        vva_real v5 = vva_real(src[r + 5]);
        int wi = wBase + ky * 6;
        vva_vec4 w0l = vva_vec4(wgt[wi]);
        vva_vec4 w0h = vva_vec4(wgt[wi + 1]);
        vva_vec4 w1l = vva_vec4(wgt[wi + 2]);
        vva_vec4 w1h = vva_vec4(wgt[wi + 3]);
        vva_vec4 w2l = vva_vec4(wgt[wi + 4]);
        vva_vec4 w2h = vva_vec4(wgt[wi + 5]);
        a0l += v0 * w0l + v1 * w1l + v2 * w2l;
        a0h += v0 * w0h + v1 * w1h + v2 * w2h;
        a1l += v1 * w0l + v2 * w1l + v3 * w2l;
        a1h += v1 * w0h + v2 * w1h + v3 * w2h;
        a2l += v2 * w0l + v3 * w1l + v4 * w2l;
        a2h += v2 * w0h + v3 * w1h + v4 * w2h;
        a3l += v3 * w0l + v4 * w1l + v5 * w2l;
        a3h += v3 * w0h + v4 * w1h + v5 * w2h;
    }
}
vva_vec4 bl = vva_vec4(wgt[P_BIAS + ocb * 2]);
vva_vec4 bh = vva_vec4(wgt[P_BIAS + ocb * 2 + 1]);
int base = (ocb * 8) * P_PLANE + (y + 1) * pw + x0 + 1;
vva_out(base, x0, a0l + bl, a0h + bh);
vva_out(base + 1, x0 + 1, a1l + bl, a1h + bh);
vva_out(base + 2, x0 + 2, a2l + bl, a2h + bh);
vva_out(base + 3, x0 + 3, a3l + bl, a3h + bh);
