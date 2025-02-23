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

#ifndef BACKEND_H_INCLUDED
#define BACKEND_H_INCLUDED

#include "config.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdlib.h>
#include <fstream>
#include <ostream>
#include <iostream>
#include <new>
#include <numeric>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <map>
#include <iterator>
#include <filesystem>
#include <stdarg.h>

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferRuntimeBase.h"
#include "NvInferSafeRuntime.h"
#include "NvInferConsistency.h"
#include "sha2.h"

#include "Utils.h"

using namespace Utils;

template <typename net_t> class BackendTRT;

#define ASSERT(condition)                                           \
    {                                                               \
        if (!(condition)) {                                         \
            myprintf_error("Assertion failure %s(%d): %s\n",        \
                __FILE__, __LINE__, #condition);                    \
            throw std::runtime_error("TensorRT error");             \
        }                                                           \
    }

#define checkCUDA(error)                                            \
    {                                                               \
        if (error != cudaSuccess) {                                 \
            myprintf_error("Error on %s(%d): %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(error));     \
            throw std::runtime_error("CUDA error");                 \
        }                                                           \
    }

class BackendContext {
public:
    bool m_buffers_allocated{false};
    // Only TENSORRT backend are used.
    std::unique_ptr<nvinfer1::IExecutionContext> mContext{nullptr};
    std::map<std::string, void*> mBuffers;
};

class BackendLayer {
public:
    unsigned int channels{0};
    unsigned int outputs{0};
    unsigned int filter_size{0};
    std::vector<void *> weights;
    bool is_input_convolution{false};
    bool is_residual_block{false};
    bool is_se_block{false};
    bool is_value{false};
    bool is_policy{false};
    // Only TENSORRT backend are used.
    std::vector<int64_t> weights_size;
    std::string name;
};

struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;

template <typename net_t>
class Backend {
public:
    Backend() {}
    Backend(
        const int gpu,
        const bool silent = false
    );

    virtual ~Backend() = default;

    void initialize(
        const NetworkType net_type,
        const size_t num_worker_threads,
        const std::string &model_hash = ""
    );

    virtual void push_input_convolution(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases
    ) = 0;

    virtual void push_residual(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights_1,
        const std::vector<float>& biases_1,
        const std::vector<float>& weights_2,
        const std::vector<float>& biases_2
    ) = 0;

    virtual void push_residual_se(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights_1,
        const std::vector<float>& biases_1,
        const std::vector<float>& weights_2,
        const std::vector<float>& biases_2,
        const std::vector<float>& se_fc1_w,
        const std::vector<float>& se_fc1_b,
        const std::vector<float>& se_fc2_w,
        const std::vector<float>& se_fc2_b
    ) = 0;

    virtual void push_convolve(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::vector<float>& weights,
        const std::vector<float>& biases,
        const std::vector<float>& ip1_w,
        const std::vector<float>& ip1_b,
        const std::vector<float>& ip2_w,
        const std::vector<float>& ip2_b
    ) = 0;

    virtual void forward_activations(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        BackendContext& cudnn_context,
        const size_t batch_size = 1
    ) = 0;

    void forward(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        const int tid,
        const size_t batch_size = 1
    );

    virtual size_t get_layer_count() const {
        return m_layers.size();
    }

    virtual bool has_fp16_compute() const {
        return m_fp16_compute;
    }

    virtual bool has_tensor_cores() const {
        return m_tensorcore;
    }

    std::vector<BackendLayer> m_layers;
    std::vector<std::unique_ptr<BackendContext>> m_context;

protected:
    bool m_fp16_compute{false};
    bool m_tensorcore{false};
    int m_num_worker_threads{1};
    cudaDeviceProp m_device_prop;
    std::string m_model_hash{""};
    NetworkType m_net_type{NetworkType::LEELA_ZERO};
};

#endif
