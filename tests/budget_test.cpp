// Memory budget helpers of the engine (included directly to reach the anonymous namespace).
#include <cstdio>

#include "../core/src/vva.cpp"

static int g_failed = 0, g_checks = 0;
#define CHECK(c, m)                                    \
    do {                                               \
        ++g_checks;                                    \
        if (!(c)) {                                    \
            ++g_failed;                                \
            std::printf("  FAIL: %s\n", m);            \
        }                                              \
    } while (0)

int main() {
    const std::string meminfo = "MemTotal:        7500000 kB\nMemFree:          300000 kB\nMemAvailable:    3200000 kB\nBuffers: 1 kB\n";
    CHECK(vva::parseMemAvailableKb(meminfo) == 3200000, "parses MemAvailable");
    CHECK(vva::parseMemAvailableKb("MemTotal: 5 kB\n") == -1, "missing line");
    CHECK(vva::parseMemAvailableKb("MemAvailable: junk kB\n") == -1, "garbage value");
    CHECK(vva::parseMemAvailableKb("") == -1, "empty text");
    const size_t mb = 1024 * 1024;
    CHECK(vva::autoBudgetBytes(-1) == 320 * mb, "unknown memory: the old default");
    CHECK(vva::autoBudgetBytes(500000) == 320 * mb, "little free memory: the floor");
    CHECK(vva::autoBudgetBytes(2000000) == 2000000ull * 1024 / 4, "a quarter of the free memory in between");
    CHECK(vva::autoBudgetBytes(2400000) > 320 * mb && vva::autoBudgetBytes(2400000) < 768 * mb, "2.4 GB free gives a mid value");
    CHECK(vva::autoBudgetBytes(3200000) == 768 * mb, "3.2 GB free (a quarter is 781 MB) is capped at the ceiling");
    CHECK(vva::autoBudgetBytes(100000000) == 768 * mb, "lots of memory: the ceiling");
    CHECK(vva::currentAutoBudget() >= 320 * mb && vva::currentAutoBudget() <= 768 * mb, "the live value is within the limits");
    std::printf("budget_test: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
