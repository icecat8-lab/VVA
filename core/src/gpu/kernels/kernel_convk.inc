int x = int(gl_GlobalInvocationID.x);
int y = int(gl_GlobalInvocationID.y) + P_YBASE;
int ocb = int(gl_GlobalInvocationID.z);
if (x >= P_W || y >= P_H) {
    return;
}
int pw = P_W + 2;
int K = P_K;
int pad = K / 2;
int taps = K * K;
vec4 al = vec4(0.0);
vec4 ah = vec4(0.0);
for (int ic = 0; ic < P_CIN; ++ic) {
    int wBase = (ocb * P_CIN + ic) * taps * 2;
    int plane0 = ic * P_PLANE;
    for (int ky = 0; ky < K; ++ky) {
        int iy = y + ky - pad;
        if (iy < 0 || iy >= P_H) {
            continue;
        }
        int rowBase = plane0 + (iy + 1) * pw + 1;
        for (int kx = 0; kx < K; ++kx) {
            int ix = x + kx - pad;
            if (ix < 0 || ix >= P_W) {
                continue;
            }
            float v = float(src[rowBase + ix]);
            int wi = wBase + (ky * K + kx) * 2;
            al += v * vec4(wgt[wi]);
            ah += v * vec4(wgt[wi + 1]);
        }
    }
}
vec4 bl = vec4(wgt[P_BIAS + ocb * 2]);
vec4 bh = vec4(wgt[P_BIAS + ocb * 2 + 1]);
int base = (ocb * 8) * P_PLANE + (y + 1) * pw + x + 1;
vva_out(base, x, al + bl, ah + bh);
