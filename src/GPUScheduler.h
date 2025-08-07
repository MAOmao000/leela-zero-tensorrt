/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Junhee Yoo and contributors
    Copyright (C) 2025 MAOmao000

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

#ifndef GPUSCHEDULER_H_INCLUDED
#define GPUSCHEDULER_H_INCLUDED
#include "config.h"

#include <list>
#include <thread>
#include <vector>

#include "ForwardPipe.h"
#include "BackendTensorRT.h"
#include "GTP.h"
#include "SMP.h"
#include "ThreadPool.h"

template <typename net_t>
class GPUScheduler : public ForwardPipe {
    class ForwardQueueEntry {
    public:
        std::mutex mutex;
        std::condition_variable cv;
        const bool full_batch;
#if defined(USE_TENSOR_FP16)
        const std::vector<__half>& in;
        std::vector<__half>& out_p;
        std::vector<__half>& out_v;
        ForwardQueueEntry(
            const std::vector<__half>& input,
            std::vector<__half>& output_pol,
            std::vector<__half>& output_val,
            const bool full)
                : full_batch(full),
                in(input),
                out_p(output_pol),
                out_v(output_val) {}
#else
        const std::vector<float>& in;
        std::vector<float>& out_p;
        std::vector<float>& out_v;
        ForwardQueueEntry(
            const std::vector<float>& input,
            std::vector<float>& output_pol,
            std::vector<float>& output_val,
            const bool full)
                : full_batch(full),
                in(input),
                out_p(output_pol),
                out_v(output_val) {}
#endif
    };

public:
    GPUScheduler();
    ~GPUScheduler() override;

    void initialize(
        const int channels,
        const NetworkType net_type,
        const std::string &model_hash = nullptr
    ) override;
    void push_weights(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::shared_ptr<const ForwardPipeWeights> weights
    ) override;
#if defined(USE_TENSOR_FP16)
    bool forward(
        const std::vector<__half>& input,
        std::vector<__half>& output_pol,
        std::vector<__half>& output_val,
        const bool full_batch
    ) override;
#else
    bool needs_autodetect() override;
    bool forward(
        const std::vector<float>& input,
        std::vector<float>& output_pol,
        std::vector<float>& output_val,
        const bool full_batch
    ) override;
#endif

private:
    void push_input_convolution(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const size_t weight_index,
        const std::shared_ptr<const ForwardPipeWeights> weights
    );
    void push_residual(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const size_t weight_index,
        const std::shared_ptr<const ForwardPipeWeights> weights
    );
    void push_residual_se(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const size_t weight_index,
        const std::shared_ptr<const ForwardPipeWeights> weights
    );
    void push_convolve(
        const unsigned int filter_size,
        const unsigned int channels,
        const unsigned int outputs,
        const std::shared_ptr<const ForwardPipeWeights> weights
    );
    void batch_worker(
        const size_t gnum,
        const size_t tid = -1
    );
    void drain() override;
    void resume() override;

    bool m_running = true;
    std::atomic<bool> m_draining{false};
    std::vector<std::unique_ptr<BackendTRT<net_t>>> m_backend;

    std::mutex m_mutex;
    std::condition_variable m_cv;

    std::list<std::shared_ptr<ForwardQueueEntry>> m_forward_queue;
    std::list<std::thread> m_worker_threads;

    NetworkType m_net_type{NetworkType::LEELA_ZERO};
};
#endif
