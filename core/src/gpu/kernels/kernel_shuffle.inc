int x = int(gl_GlobalInvocationID.x);
int y = int(gl_GlobalInvocationID.y);
int c = int(gl_GlobalInvocationID.z);
int ow = P_W * P_R;
int oh = P_H * P_R;
if (x >= ow || y >= oh || c >= P_COUT) {
    return;
}
int ic = c * P_R * P_R + (y % P_R) * P_R + (x % P_R);
dst[c * P_PLANEOUT + (y + 1) * (ow + 2) + x + 1] = src[ic * P_PLANEIN + (y / P_R + 1) * (P_W + 2) + (x / P_R + 1)];
