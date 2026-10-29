// Test helper: runs the engine's classifier resize on a PPM image and dumps the RGB bytes.
//   resize_test in.ppm SIZE out.rgb
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../core/src/vva.cpp"

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 1;
    int w = 0, h = 0, maxval = 0;
    if (std::fscanf(f, "P6 %d %d %d", &w, &h, &maxval) != 3 || maxval != 255) return 1;
    std::fgetc(f);
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3), rgba(static_cast<size_t>(w) * h * 4);
    if (std::fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) return 1;
    std::fclose(f);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        rgba[4 * i] = rgb[3 * i];
        rgba[4 * i + 1] = rgb[3 * i + 1];
        rgba[4 * i + 2] = rgb[3 * i + 2];
        rgba[4 * i + 3] = 255;
    }
    std::vector<uint8_t> out;
    vva::resizeRgb8(rgba.data(), w, h, std::atoi(argv[2]), out);
    f = std::fopen(argv[3], "wb");
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    return 0;
}
