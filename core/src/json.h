// Minimal, dependency-free JSON parser used for the .vva metadata block.
// Strict enough for untrusted input: bounded nesting depth, no recursion
// beyond kMaxDepth, every failure reports a message instead of crashing.
#ifndef VVA_JSON_H
#define VVA_JSON_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace vva {
namespace json {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::vector<std::string> keys;  // Object member names, same order as `members`
    std::vector<Value> members;     // Object member values

    bool isNull() const { return type == Type::Null; }
    bool isBool() const { return type == Type::Bool; }
    bool isNumber() const { return type == Type::Number; }
    bool isString() const { return type == Type::String; }
    bool isArray() const { return type == Type::Array; }
    bool isObject() const { return type == Type::Object; }

    // Returns the first member with this key, or nullptr.
    const Value* find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] == key) return &members[i];
        }
        return nullptr;
    }
};

class Parser {
public:
    Parser(const char* data, size_t size) : begin_(data), p_(data), end_(data + size) {}

    bool parse(Value& out, std::string& error) {
        skipWhitespace();
        if (!parseValue(out, 0)) {
            error = error_;
            return false;
        }
        skipWhitespace();
        if (p_ != end_) {
            fail("unexpected characters after the JSON value");
            error = error_;
            return false;
        }
        return true;
    }

private:
    static constexpr int kMaxDepth = 64;

    const char* begin_;
    const char* p_;
    const char* end_;
    std::string error_;

    bool fail(const std::string& message) {
        if (error_.empty()) {
            error_ = "JSON error at byte " + std::to_string(p_ - begin_) + ": " + message;
        }
        return false;
    }

    void skipWhitespace() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }

    bool consumeLiteral(const char* literal) {
        const char* q = p_;
        for (const char* l = literal; *l; ++l, ++q) {
            if (q >= end_ || *q != *l) return false;
        }
        p_ = q;
        return true;
    }

    bool parseValue(Value& out, int depth) {
        if (depth > kMaxDepth) return fail("nesting is too deep");
        if (p_ >= end_) return fail("unexpected end of input");

        switch (*p_) {
            case '{': return parseObject(out, depth);
            case '[': return parseArray(out, depth);
            case '"':
                out.type = Value::Type::String;
                return parseString(out.string);
            case 't':
                if (!consumeLiteral("true")) return fail("invalid literal");
                out.type = Value::Type::Bool;
                out.boolean = true;
                return true;
            case 'f':
                if (!consumeLiteral("false")) return fail("invalid literal");
                out.type = Value::Type::Bool;
                out.boolean = false;
                return true;
            case 'n':
                if (!consumeLiteral("null")) return fail("invalid literal");
                out.type = Value::Type::Null;
                return true;
            default:
                if (*p_ == '-' || (*p_ >= '0' && *p_ <= '9')) return parseNumber(out);
                return fail("unexpected character");
        }
    }

    bool parseObject(Value& out, int depth) {
        out.type = Value::Type::Object;
        ++p_;  // '{'
        skipWhitespace();
        if (p_ < end_ && *p_ == '}') {
            ++p_;
            return true;
        }
        for (;;) {
            skipWhitespace();
            if (p_ >= end_ || *p_ != '"') return fail("expected a string key");
            std::string key;
            if (!parseString(key)) return false;
            skipWhitespace();
            if (p_ >= end_ || *p_ != ':') return fail("expected ':' after key");
            ++p_;
            skipWhitespace();
            Value member;
            if (!parseValue(member, depth + 1)) return false;
            out.keys.push_back(std::move(key));
            out.members.push_back(std::move(member));
            skipWhitespace();
            if (p_ >= end_) return fail("unterminated object");
            if (*p_ == ',') {
                ++p_;
                continue;
            }
            if (*p_ == '}') {
                ++p_;
                return true;
            }
            return fail("expected ',' or '}'");
        }
    }

    bool parseArray(Value& out, int depth) {
        out.type = Value::Type::Array;
        ++p_;  // '['
        skipWhitespace();
        if (p_ < end_ && *p_ == ']') {
            ++p_;
            return true;
        }
        for (;;) {
            skipWhitespace();
            Value element;
            if (!parseValue(element, depth + 1)) return false;
            out.array.push_back(std::move(element));
            skipWhitespace();
            if (p_ >= end_) return fail("unterminated array");
            if (*p_ == ',') {
                ++p_;
                continue;
            }
            if (*p_ == ']') {
                ++p_;
                return true;
            }
            return fail("expected ',' or ']'");
        }
    }

    static int hexValue(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    bool parseHex4(uint32_t& out) {
        if (end_ - p_ < 4) return fail("truncated \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            int h = hexValue(p_[i]);
            if (h < 0) return fail("invalid \\u escape");
            v = (v << 4) | static_cast<uint32_t>(h);
        }
        p_ += 4;
        out = v;
        return true;
    }

    static void appendUtf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    bool parseString(std::string& out) {
        ++p_;  // opening quote
        out.clear();
        for (;;) {
            if (p_ >= end_) return fail("unterminated string");
            unsigned char c = static_cast<unsigned char>(*p_++);
            if (c == '"') return true;
            if (c < 0x20) return fail("control character inside string");
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (p_ >= end_) return fail("unterminated escape");
            char e = *p_++;
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!parseHex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (end_ - p_ < 2 || p_[0] != '\\' || p_[1] != 'u') {
                            return fail("unpaired surrogate in \\u escape");
                        }
                        p_ += 2;
                        uint32_t low = 0;
                        if (!parseHex4(low)) return false;
                        if (low < 0xDC00 || low > 0xDFFF) return fail("invalid surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("unpaired surrogate in \\u escape");
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default:
                    return fail("invalid escape sequence");
            }
        }
    }

    bool parseNumber(Value& out) {
        const char* start = p_;
        while (p_ < end_ && ((*p_ >= '0' && *p_ <= '9') || *p_ == '-' || *p_ == '+' || *p_ == '.' ||
                             *p_ == 'e' || *p_ == 'E')) {
            ++p_;
        }
        const size_t len = static_cast<size_t>(p_ - start);
        if (len == 0 || len > 64) return fail("invalid number");

        // Locale-independent parsing (an app may have changed the C locale).
        std::istringstream stream(std::string(start, len));
        stream.imbue(std::locale::classic());
        double value = 0.0;
        stream >> value;
        if (stream.fail() || !stream.eof() || !std::isfinite(value)) {
            p_ = start;
            return fail("invalid number");
        }
        out.type = Value::Type::Number;
        out.number = value;
        return true;
    }
};

inline bool parse(const char* data, size_t size, Value& out, std::string& error) {
    Parser parser(data, size);
    return parser.parse(out, error);
}

}  // namespace json
}  // namespace vva

#endif  // VVA_JSON_H
