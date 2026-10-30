// Shared helpers, included at global scope by every GPU kernel and by the host emulator.
float vva_act(float v, float alpha) {
    return v > 0.0 ? v : v * alpha;
}
