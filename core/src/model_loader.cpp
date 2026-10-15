#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <string>

namespace vva {

struct LayerMeta {
    std::string name;
    std::string precision;
    size_t offset;
    size_t length;
};

class ModelLoader {
private:
    int fd = -1;
    size_t file_size = 0;
    void* mapped_data = nullptr;
    std::vector<LayerMeta> layers;

public:
    bool loadVVA(const std::string& filepath) {
        fd = open(filepath.c_str(), O_RDONLY);
        if (fd < 0) return false;

        struct stat sb;
        if (fstat(fd, &sb) == -1) return false;
        file_size = sb.st_size;

        mapped_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_data == MAP_FAILED) return false;

        uint8_t* ptr = static_cast<uint8_t*>(mapped_data);
        
        if (memcmp(ptr, "VVA1", 4) != 0) return false;
        ptr += 4;

        uint32_t json_size = *reinterpret_cast<uint32_t*>(ptr);
        ptr += 4;

        std::string json_str(reinterpret_cast<char*>(ptr), json_size);
        ptr += json_size;

        return true;
    }

    ~ModelLoader() {
        if (mapped_data != MAP_FAILED) munmap(mapped_data, file_size);
        if (fd >= 0) close(fd);
    }
};

}
