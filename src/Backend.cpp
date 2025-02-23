/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Henrik Forsten
    Copyright (C) 2024 MAOmao000

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
*/

#include "config.h"

#include "Backend.h"
#include "GTP.h"

using namespace Utils;

template <typename net_t>
Backend<net_t>::Backend(
    const int gpu,
    const bool silent) {

    auto best_bandwidth = 0.0;
    auto found_device = false;
    auto nDevices = 0;
    auto best_device_id = 0;
    cudaDeviceProp best_device;

    cudaGetDeviceCount(&nDevices);

    if (!silent) {
        myprintf("Detected %d CUDA devices.\n", nDevices);
    }

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        auto bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        if (!silent) {
            myprintf("Device Number: %d\n", i);
            myprintf("  Device name: %s\n", prop.name);
            myprintf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            myprintf("  Peak Memory Bandwidth (GB/s): %.1f\n\n", bandwidth);
        }

        bool preferred = (gpu == i);

        if (bandwidth > best_bandwidth || preferred) {
            best_bandwidth = bandwidth;
            best_device = prop;
            best_device_id = i;
            if (preferred) {
                best_bandwidth = std::numeric_limits<decltype(best_bandwidth)>::max();
            } else {
                best_bandwidth = bandwidth;
            }
            found_device = true;
        }
    }

    if (!found_device) {
        myprintf("No suitable CUDA device found.\n");
        exit(EXIT_FAILURE);
    }

    myprintf("Selected device: %s\n", best_device.name);
    myprintf("with compute capability %d.%d.\n", best_device.major, best_device.minor);

    if (best_device.major >= 7) {
        m_tensorcore = true;
    } else if (best_device.major >= 6) {
        m_fp16_compute = true;
    }

    cudaSetDevice(best_device_id);
    m_device_prop = best_device;
}

template <typename net_t>
void Backend<net_t>::initialize(
    const NetworkType net_type,
    const size_t num_worker_threads,
    const std::string &model_hash) {

    m_net_type = net_type;
    m_num_worker_threads = static_cast<int>(num_worker_threads);
    m_model_hash = model_hash;
}

template <typename net_t>
void Backend<net_t>::forward(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    const int tid,
    const size_t batch_size) {

    forward_activations(input, output_pol, output_val, *m_context[tid], batch_size);
}

template class Backend<float>;
template class Backend<half_float::half>;
