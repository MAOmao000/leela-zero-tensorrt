/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"

#include <boost/filesystem.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <cstdarg>
#include <cstdio>
#include <mutex>

#include "Utils.h"

#include "GTP.h"

Utils::ThreadPool thread_pool;

auto constexpr z_entries = 1000;
std::array<float, z_entries> z_lookup;

void Utils::create_z_table() {
    for (auto i = 1; i < z_entries + 1; i++) {
        boost::math::students_t dist(i);
        auto z =
            boost::math::quantile(boost::math::complement(dist, cfg_ci_alpha));
        z_lookup[i - 1] = z;
    }
}

float Utils::cached_t_quantile(const int v) {
    if (v < 1) {
        return z_lookup[0];
    }
    if (v < z_entries) {
        return z_lookup[v - 1];
    }
    // z approaches constant when v is high enough.
    // With default lookup table size the function is flat enough that we
    // can just return the last entry for all v bigger than it.
    return z_lookup[z_entries - 1];
}

bool Utils::input_pending() {
#ifdef HAVE_SELECT
    fd_set read_fds;
    FD_ZERO(&read_fds);
    FD_SET(0, &read_fds);
    struct timeval timeout{0, 0};
    select(1, &read_fds, nullptr, nullptr, &timeout);
    return FD_ISSET(0, &read_fds);
#else
    static int init = 0, pipe;
    static HANDLE inh;
    DWORD dw;

    if (!init) {
        init = 1;
        inh = GetStdHandle(STD_INPUT_HANDLE);
        pipe = !GetConsoleMode(inh, &dw);
        if (!pipe) {
            SetConsoleMode(inh,
                           dw & ~(ENABLE_MOUSE_INPUT | ENABLE_WINDOW_INPUT));
            FlushConsoleInputBuffer(inh);
        }
    }

    if (pipe) {
        if (!PeekNamedPipe(inh, nullptr, 0, nullptr, &dw, nullptr)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        return dw;
    } else {
        if (!GetNumberOfConsoleInputEvents(inh, &dw)) {
            myprintf("Nothing at other end - exiting\n");
            exit(EXIT_FAILURE);
        }

        return dw > 1;
    }
    return false;
#endif
}

static std::mutex IOmutex;

static void myprintf_base(const char* const fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);

    vfprintf(stderr, fmt, ap);

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        vfprintf(cfg_logfile_handle, fmt, ap2);
    }
    va_end(ap2);
}

void Utils::myprintf(const char* const fmt, ...) {
    if (cfg_quiet) {
        return;
    }

    va_list ap;
    va_start(ap, fmt);
    myprintf_base(fmt, ap);
    va_end(ap);
}

void Utils::myprintf_error(const char* const fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    myprintf_base(fmt, ap);
    va_end(ap);
}

static void gtp_fprintf(FILE* const file, const std::string& prefix,
                        const char* const fmt, va_list ap) {
    fprintf(file, "%s ", prefix.c_str());
    vfprintf(file, fmt, ap);
    fprintf(file, "\n\n");
}

static void gtp_base_printf(const int id, std::string prefix,
                            const char* const fmt, va_list ap) {
    if (id != -1) {
        prefix += std::to_string(id);
    }
    gtp_fprintf(stdout, prefix, fmt, ap);
    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        gtp_fprintf(cfg_logfile_handle, prefix, fmt, ap);
    }
}

void Utils::gtp_printf(const int id, const char* const fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    gtp_base_printf(id, "=", fmt, ap);
    va_end(ap);
}

void Utils::gtp_printf_raw(const char* const fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);

    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        va_start(ap, fmt);
        vfprintf(cfg_logfile_handle, fmt, ap);
        va_end(ap);
    }
}

void Utils::gtp_fail_printf(const int id, const char* const fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    gtp_base_printf(id, "?", fmt, ap);
    va_end(ap);
}

void Utils::log_input(const std::string& input) {
    if (cfg_logfile_handle) {
        std::lock_guard<std::mutex> lock(IOmutex);
        fprintf(cfg_logfile_handle, ">>%s\n", input.c_str());
    }
}

size_t Utils::ceilMultiple(const size_t a, const size_t b) {
    if (a % b == 0) {
        return a;
    }

    auto ret = a + (b - a % b);
    return ret;
}

std::string Utils::leelaz_file(const std::string& file) {
#if defined(_WIN32) || defined(__ANDROID__)
    boost::filesystem::path dir(boost::filesystem::current_path());
#else
    // https://stackoverflow.com/a/26696759
    const char* homedir;
    if ((homedir = getenv("HOME")) == nullptr) {
        struct passwd* pwd;
        // NOLINTNEXTLINE(runtime/threadsafe_fn)
        if ((pwd = getpwuid(getuid())) == nullptr) {
            return std::string();
        }
        homedir = pwd->pw_dir;
    }
    boost::filesystem::path dir(homedir);
    dir /= ".local/share/leela-zero";
#endif
    boost::filesystem::create_directories(dir);
    dir /= file;
    return dir.string();
}

std::vector<float> Utils::softmax(const std::vector<float>& input,
                                  const float temperature) {

    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(cbegin(input), cend(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha) / temperature);
        denom += val;
        output.push_back(val);
    }

    for (auto& out : output) {
        out /= denom;
    }

    return output;
}

#ifdef _WIN32
HANDLE Utils::lockFile(const std::string& file) {
    HANDLE hFile = CreateFile(
        file.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    OVERLAPPED overlapped = {0};
    auto locked = LockFileEx(
        hFile,
        LOCKFILE_EXCLUSIVE_LOCK,
        0, // reserved
        0, // number of bytes to lock
        0, // offset high
        &overlapped);
    if (!locked) {
        CloseHandle(hFile);
        return nullptr;
    }
    return hFile;
#else
int Utils::lockFile(const std::string& file) {
    int fd = open(file.c_str(), O_RDWR | O_CREAT, 0664);
    if (fd == -1) {
        return -1;
    }
    struct flock fl;
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0; // 0 means to lock the whole file
    if (fcntl(fd, F_SETLKW, &fl) == -1) {
        close(fd);
        return -1;
    }
    return fd;
#endif
}

#ifdef _WIN32
void Utils::unlockFile(HANDLE hFile) {
    OVERLAPPED overlapped = {0};
    UnlockFileEx(hFile, 0, 0, 0, &overlapped);
    CloseHandle(hFile);
#else
void Utils::unlockFile(int fd) {
    struct flock fl;
    fl.l_type = F_UNLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 0;
    if (fcntl(fd, F_SETLK, &fl) == -1) {
        myprintf_error("Utils::unlockFile fcntl error\n");
    }
    close(fd);
#endif
}
