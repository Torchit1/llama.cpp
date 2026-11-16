#include "ggml-impl.h"

#include <cstdlib>
#include <exception>

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();

// see ggml_live_switch() in ggml.h
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <sys/stat.h>

int ggml_live_switch(const char * name, int def) {
    static std::mutex                 mtx;
    static std::map<std::string, int> file_vals;
    static const char *               path       = getenv("GGML_SWITCH_FILE");
    static long long                  last_check = -1;
    static long long                  last_mtime = -1;
    // Answers by the address of the name (callers pass literals), valid until the file changes: a kernel dispatch
    // asks several switches per launch, and a string lookup plus getenv() for each adds up to milliseconds a step.
    struct answer {
        const char * name;
        int          def;
        int          val;
    };
    static constexpr int n_slots = 256;
    static answer        slots[n_slots];

    std::lock_guard<std::mutex> lock(mtx);
    if (path) {
        const long long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (last_check < 0 || now - last_check >= 10) {
            last_check = now;
            struct stat st;
            const long long mt = stat(path, &st) == 0 ? (long long) st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec : 0;
            if (mt != last_mtime) {
                last_mtime = mt;
                file_vals.clear();
                for (auto & a : slots) {
                    a.name = nullptr;
                }
                if (FILE * f = fopen(path, "r")) {
                    char line[256];
                    while (fgets(line, sizeof(line), f)) {
                        char * eq = strchr(line, '=');
                        if (eq) {
                            *eq = 0;
                            file_vals[line] = atoi(eq + 1);
                        }
                    }
                    fclose(f);
                }
            }
        }
    }
    answer & a = slots[((uintptr_t) name >> 3) % n_slots];
    if (a.name == name && a.def == def) {
        return a.val;
    }
    int val = def;
    bool found = false;
    if (path) {
        const auto it = file_vals.find(name);
        if (it != file_vals.end()) {
            val   = it->second;
            found = true;
        }
    }
    if (!found) {
        const char * e = getenv(name);
        val = e ? atoi(e) : def;
    }
    a = { name, def, val };
    return val;
}
