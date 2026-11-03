// 3x3 convolution computing 4 pixels x 4 output channels per invocation (fewer registers, more invocations).
// The 8-channel weight blocks are shared with the other variants: z = 2 * block + half.
int x0 = int(gl_GlobalInvocationID.x) * 4;
int y = int(gl_GlobalInvocationID.y) + P_YBASE;
int g = int(gl_GlobalInvocationID.z);
int ocb = g / 2;
int half = g - ocb * 2;
if (x0 >= P_W || y >= P_H) {
    return;
}
int pw = P_W + 2;
vec4 a0 = vec4(0.0);
vec4 a1 = vec4(0.0);
vec4 a2 = vec4(0.0);
vec4 a3 = vec4(0.0);
for (int ic = 0; ic < P_CIN; ++ic) {
    int wBase = (ocb * P_CIN + ic) * 18 + half;
    int rBase = ic * P_PLANE + y * pw + x0;
    for (int ky = 0; ky < 3; ++ky) {
        int r = rBase + ky * pw;
        float v0 = src[r];
        float v1 = src[r + 1];
        float v2 = src[r + 2];
        float v3 = src[r + 3];
        float v4 = src[r + 4];
        float v5 = src[r + 5];
        int wi = wBase + ky * 6;
        vec4 w0 = wgt[wi];
        vec4 w1 = wgt[wi + 2];
        vec4 w2 = wgt[wi + 4];
        a0 += v0 * w0 + v1 * w1 + v2 * w2;
        a1 += v1 * w0 + v2 * w1 + v3 * w2;
        a2 += v2 * w0 + v3 * w1 + v4 * w2;
        a3 += v3 * w0 + v4 * w1 + v5 * w2;
    }
}
vec4 b = wgt[P_BIAS + ocb * 2 + half];
int base = (ocb * 8 + half * 4) * P_PLANE + (y + 1) * pw + x0 + 1;
vva_out4(base, x0, a0 + b);
vva_out4(base + 1, x0 + 1, a1 + b);
vva_out4(base + 2, x0 + 2, a2 + b);
vva_out4(base + 3, x0 + 3, a3 + b);
