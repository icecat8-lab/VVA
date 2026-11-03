// 3x3 convolution computing 8 pixels x 8 output channels per invocation (fewer weight loads per multiply-add).
// Same data layout and epilogue as kernel_conv3x3.inc.
int x0 = int(gl_GlobalInvocationID.x) * 8;
int y = int(gl_GlobalInvocationID.y) + P_YBASE;
int ocb = int(gl_GlobalInvocationID.z);
if (x0 >= P_W || y >= P_H) {
    return;
}
int pw = P_W + 2;
vec4 a0l = vec4(0.0);
vec4 a0h = vec4(0.0);
vec4 a1l = vec4(0.0);
vec4 a1h = vec4(0.0);
vec4 a2l = vec4(0.0);
vec4 a2h = vec4(0.0);
vec4 a3l = vec4(0.0);
vec4 a3h = vec4(0.0);
vec4 a4l = vec4(0.0);
vec4 a4h = vec4(0.0);
vec4 a5l = vec4(0.0);
vec4 a5h = vec4(0.0);
vec4 a6l = vec4(0.0);
vec4 a6h = vec4(0.0);
vec4 a7l = vec4(0.0);
vec4 a7h = vec4(0.0);
for (int ic = 0; ic < P_CIN; ++ic) {
    int wBase = (ocb * P_CIN + ic) * 18;
    int rBase = ic * P_PLANE + y * pw + x0;
    for (int ky = 0; ky < 3; ++ky) {
        int r = rBase + ky * pw;
        float v0 = src[r + 0];
        float v1 = src[r + 1];
        float v2 = src[r + 2];
        float v3 = src[r + 3];
        float v4 = src[r + 4];
        float v5 = src[r + 5];
        float v6 = src[r + 6];
        float v7 = src[r + 7];
        float v8 = src[r + 8];
        float v9 = src[r + 9];
        int wi = wBase + ky * 6;
        vec4 w0l = wgt[wi];
        vec4 w0h = wgt[wi + 1];
        vec4 w1l = wgt[wi + 2];
        vec4 w1h = wgt[wi + 3];
        vec4 w2l = wgt[wi + 4];
        vec4 w2h = wgt[wi + 5];
        a0l += v0 * w0l + v1 * w1l + v2 * w2l;
        a0h += v0 * w0h + v1 * w1h + v2 * w2h;
        a1l += v1 * w0l + v2 * w1l + v3 * w2l;
        a1h += v1 * w0h + v2 * w1h + v3 * w2h;
        a2l += v2 * w0l + v3 * w1l + v4 * w2l;
        a2h += v2 * w0h + v3 * w1h + v4 * w2h;
        a3l += v3 * w0l + v4 * w1l + v5 * w2l;
        a3h += v3 * w0h + v4 * w1h + v5 * w2h;
        a4l += v4 * w0l + v5 * w1l + v6 * w2l;
        a4h += v4 * w0h + v5 * w1h + v6 * w2h;
        a5l += v5 * w0l + v6 * w1l + v7 * w2l;
        a5h += v5 * w0h + v6 * w1h + v7 * w2h;
        a6l += v6 * w0l + v7 * w1l + v8 * w2l;
        a6h += v6 * w0h + v7 * w1h + v8 * w2h;
        a7l += v7 * w0l + v8 * w1l + v9 * w2l;
        a7h += v7 * w0h + v8 * w1h + v9 * w2h;
    }
}
vec4 bl = wgt[P_BIAS + ocb * 2];
vec4 bh = wgt[P_BIAS + ocb * 2 + 1];
int base = (ocb * 8) * P_PLANE + (y + 1) * pw + x0 + 1;
vva_out(base + 0, x0 + 0, a0l + bl, a0h + bh);
vva_out(base + 1, x0 + 1, a1l + bl, a1h + bh);
vva_out(base + 2, x0 + 2, a2l + bl, a2h + bh);
vva_out(base + 3, x0 + 3, a3l + bl, a3h + bh);
vva_out(base + 4, x0 + 4, a4l + bl, a4h + bh);
vva_out(base + 5, x0 + 5, a5l + bl, a5h + bh);
vva_out(base + 6, x0 + 6, a6l + bl, a6h + bh);
vva_out(base + 7, x0 + 7, a7l + bl, a7h + bh);
