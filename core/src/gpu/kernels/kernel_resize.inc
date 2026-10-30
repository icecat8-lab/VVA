// Nearest-neighbour upscaling by integer factors; writes the interior only (borders stay zero).
int x = int(gl_GlobalInvocationID.x);
int y = int(gl_GlobalInvocationID.y);
int c = int(gl_GlobalInvocationID.z);
int ow = P_W * P_SW;
int oh = P_H * P_SH;
if (x >= ow || y >= oh) {
    return;
}
dst[c * P_PLANEOUT + (y + 1) * (ow + 2) + x + 1] = src[c * P_PLANEIN + (y / P_SH + 1) * (P_W + 2) + (x / P_SW + 1)];
