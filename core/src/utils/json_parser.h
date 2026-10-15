#ifndef VVA_JSON_PARSER_H
#define VVA_JSON_PARSER_H

#include <string>
#include <vector>

namespace vva {

struct ParsedLayer {
    std::string name;
    std::string precision;
    std::vector<int> shape;
    size_t offset;
    size_t length;
};

class SimpleJSONParser {
public:
    static std::vector<ParsedLayer> parseMetadata(const std::string& json_str) {
        std::vector<ParsedLayer> layers;
        size_t pos = 0;
        
        while ((pos = json_str.find("\"name\":", pos)) != std::string::npos) {
            ParsedLayer layer;
            
            size_t start = json_str.find("\"", pos + 7) + 1;
            size_t end = json_str.find("\"", start);
            layer.name = json_str.substr(start, end - start);
            
            pos = json_str.find("\"precision\":", end);
            start = json_str.find("\"", pos + 12) + 1;
            end = json_str.find("\"", start);
            layer.precision = json_str.substr(start, end - start);

            pos = json_str.find("\"offset\":", end);
            start = pos + 9;
            end = json_str.find(",", start);
            layer.offset = std::stoull(json_str.substr(start, end - start));

            pos = json_str.find("\"length\":", end);
            start = pos + 9;
            end = json_str.find("}", start);
            layer.length = std::stoull(json_str.substr(start, end - start));

            layers.push_back(layer);
        }
        return layers;
    }
};

}
#endif
