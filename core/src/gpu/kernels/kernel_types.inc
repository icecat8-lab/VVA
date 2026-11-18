#ifdef VVA_STORE_F16
#define vva_store_t float16_t
#define vva_wt_t f16vec4
#else
#define vva_store_t float
#define vva_wt_t vec4
#endif
#define vva_real float
#define vva_vec4 vec4
