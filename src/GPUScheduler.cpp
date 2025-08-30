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
#include "config.h"

#if defined(USE_TENSOR_RT)

#include "GPUScheduler.h"
#include "BackendTensorRT.h"
#include "Network.h"
#include "Random.h"
#include "Utils.h"

template <typename net_t>
GPUScheduler<net_t>::GPUScheduler()
{
    // multi-gpu?
    auto gpus = cfg_gpus;
    // An empty GPU list from the command line represents autodetect.
    // Put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }

    auto silent{false};
    for (auto gpu : gpus) {
        auto net = std::make_unique<BackendTRT<net_t>>(gpu, silent);
        m_backend.emplace_back(std::move(net));
        // Starting next GPU, let's not dump full list of GPUs.
        silent = true;
    }
}

template <typename net_t>
void GPUScheduler<net_t>::initialize(
    const int channels,
    const NetworkType net_type,
    const std::string &model_hash)
{
    // For compatibility with OpenCL implementation
    (void) channels;

    m_net_type = net_type;
    // Launch the worker threads.  Minimum 1 worker per GPU, but use enough
    // threads so that we can at least concurrently schedule something to the
    // GPU.
    auto num_worker_threads =
        cfg_num_threads / cfg_batch_size / (m_backend.size() + 1) + 1;
    auto gnum = 0;
    for (auto& backend : m_backend) {
        backend->initialize(net_type, num_worker_threads, model_hash);

        for (auto i = unsigned{0}; i < num_worker_threads; i++) {
            auto t = std::thread(&GPUScheduler<net_t>::batch_worker, this, gnum, i);
            m_worker_threads.push_back(std::move(t));
        }
        gnum++;
    }
}

template <typename net_t>
GPUScheduler<net_t>::~GPUScheduler()
{
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_running = false;
    }
    m_cv.notify_all();
    for (auto& x : m_worker_threads) {
        x.join();
    }
    for (const auto& backend : m_backend) {
        for (auto iter = std::begin(backend->m_layers);
            iter != std::end(backend->m_layers);
            iter++)
        {
            const auto& layer = *iter;
            for (auto it = layer.weights.begin();
                it != layer.weights.end();
                ++it)
            {
                void *w_mem;
                cudaHostGetDevicePointer((void**)&w_mem, *it, 0);
                if (w_mem) {
                    cudaFreeAsync(w_mem, cudaStreamDefault);
                }
                cudaFreeHost(*it);
            }
        }
    }
    for (const auto& backend : m_backend) {
        for (const auto& context : backend->m_context) {
            if (context->m_buffers_allocated) {
                for (auto ptr: context->mBuffers) {
                    cudaFreeAsync(ptr.second, cudaStreamDefault);
                }
            }
        }
    }
    cudaStreamSynchronize(cudaStreamDefault);
    for (auto& backend : m_backend) {
        backend.release();
    }
}

template <typename net_t>
void GPUScheduler<net_t>::push_input_convolution(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
    for (const auto& backend : m_backend) {
        backend->push_input_convolution(
            filter_size,
            channels,
            outputs,
            weights->m_conv_weights[weight_index],
            weights->m_batchnorm_means[weight_index]
        );
    }
}

template <typename net_t>
void GPUScheduler<net_t>::push_residual(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
    for (const auto& backend : m_backend) {
        backend->push_residual(
            filter_size,
            channels,
            outputs,
            weights->m_conv_weights[weight_index],
            weights->m_batchnorm_means[weight_index],
            weights->m_conv_weights[weight_index + 1],
            weights->m_batchnorm_means[weight_index + 1]
        );
    }
}

template <typename net_t>
void GPUScheduler<net_t>::push_residual_se(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
    for (const auto& backend : m_backend) {
        backend->push_residual_se(
            filter_size,
            channels,
            outputs,
            weights->m_conv_weights[weight_index],
            weights->m_batchnorm_means[weight_index],
            weights->m_conv_weights[weight_index + 1],
            weights->m_batchnorm_means[weight_index + 1],
            weights->m_se_weights[weight_index - 1],
            weights->m_se_biases[weight_index - 1],
            weights->m_se_weights[weight_index],
            weights->m_se_biases[weight_index]
        );
    }
}

template <typename net_t>
void GPUScheduler<net_t>::push_convolve(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
    for (const auto& backend : m_backend) {
        if (outputs == Network::OUTPUTS_POLICY) {
            backend->push_convolve(
                filter_size,
                channels,
                outputs,
                weights->m_conv_pol_w,
                weights->m_bn_pol_w1,
                weights->m_ip_pol_w,
                weights->m_ip_pol_b,
                weights->m_ip_pol_w,
                weights->m_ip_pol_b
            );
        } else {
            backend->push_convolve(
                filter_size,
                channels,
                outputs,
                weights->m_conv_val_w,
                weights->m_bn_val_w1,
                weights->m_ip1_val_w,
                weights->m_ip1_val_b,
                weights->m_ip2_val_w,
                weights->m_ip2_val_b
            );
        }
    }
}

template <typename net_t>
void GPUScheduler<net_t>::push_weights(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const std::shared_ptr<const ForwardPipeWeights> weights)
{
    auto weight_index = size_t{0};
    push_input_convolution(
        filter_size,
        channels,
        outputs,
        weight_index,
        weights
    );
    weight_index++;
    if (m_net_type == NetworkType::LEELA_ZERO) {
        // residual blocks : except the first entry,
        // the second ~ last entry is all on residual topwer
        for (auto i = size_t{0}; i < weights->m_conv_weights.size() / 2; i++) {
            push_residual(
                filter_size,
                outputs,
                outputs,
                weight_index,
                weights
            );
            weight_index += 2;
        }
    } else if (m_net_type == NetworkType::MINIGO_SE) {
        // residual blocks : except the first entry,
        // the second ~ last entry is all on residual topwer
        for (auto i = size_t{0}; i < weights->m_conv_weights.size() / 2; i++) {
            push_residual_se(
                filter_size,
                outputs,
                outputs,
                weight_index,
                weights
            );
            weight_index += 2;
        }
    }
    // Output head convolutions
    push_convolve(
        1,
        outputs,
        Network::OUTPUTS_POLICY,
        weights
    );
    push_convolve(
        1,
        outputs,
        Network::OUTPUTS_VALUE,
        weights
    );
    // Asynchronously cudaMemcpyAsync
    cudaStreamSynchronize(cudaStreamPerThread);
}

template <typename net_t>
bool GPUScheduler<net_t>::forward(
    const std::vector<float>& input,
    std::vector<float>& output_pol,
    std::vector<float>& output_val,
    const bool full_batch)
{
    if (m_draining.load()) {
        return false;
    }

    auto entry =
        std::make_shared<ForwardQueueEntry>(input, output_pol, output_val, full_batch);
    size_t queue_size = 0;
    std::unique_lock<std::mutex> lk(entry->mutex);
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.emplace_back(entry);
        queue_size = m_forward_queue.size();
    }
    if (!full_batch || queue_size >= cfg_batch_size) {
        m_cv.notify_one();
    }
    entry->cv.wait(lk);
    if (output_pol[0] == -1.0f) {
        return false;
    }
    return true;
}

template <typename net_t>
void GPUScheduler<net_t>::batch_worker(
    const size_t gnum,
    const size_t tid)
{
    constexpr auto in_size = Network::INPUT_CHANNELS * NUM_INTERSECTIONS;
    constexpr auto out_pol_size = POTENTIAL_MOVES;
    constexpr auto out_val_size = 1;
    // Returns the batch picked up from the queue (m_forward_queue)
    auto pickup_task = [this]() {

        std::list<std::shared_ptr<ForwardQueueEntry>> inputs;
        std::unique_lock<std::mutex> lk(m_mutex);
        m_cv.wait(lk, [this] {
            return !m_running ||
                m_draining.load() ||
                m_forward_queue.size() >= cfg_batch_size ||
                (m_forward_queue.size() == 1 && !m_forward_queue.front()->full_batch);
        });
        // first:  m_running:true  m_draining:false m_forward_queue.size:1
        // second: m_running:true  m_draining:false m_forward_queue.size:10
        // next:   m_running:true  m_draining:false m_forward_queue.size:10
        // drain:  m_running:true  m_draining:true  m_forward_queue.size:0-10
        // next:   m_running:true  m_draining:true  m_forward_queue.size:0
        // quit:   m_running:false m_draining:false m_forward_queue.size:0
        if (!m_running) {
            return inputs;
        }
        auto count = m_forward_queue.size();
        if (!count) {
            return inputs;
        } else if (count >= static_cast<size_t>(cfg_batch_size)) {
            count = cfg_batch_size;
        } else if (!m_draining.load() &&
            m_forward_queue.front()->full_batch) {
            return inputs;
        }
        // Move 'count' evals from shared queue to local list.
        auto end = begin(m_forward_queue);
        std::advance(end, count);
        std::move(begin(m_forward_queue), end, std::back_inserter(inputs));
        m_forward_queue.erase(begin(m_forward_queue), end);
        return inputs;
    };
    auto batch_input = std::vector<float>(in_size * cfg_batch_size);
    auto batch_output_pol = std::vector<float>(out_pol_size * cfg_batch_size);
    auto batch_output_val = std::vector<float>(out_val_size * cfg_batch_size);
    while (true) {
        auto inputs = pickup_task();

        if (!m_running) {
            return;
        }
        auto count = inputs.size();
        if (!count) {
            continue;
        }
        // prepare input for forward() call
        batch_input.resize(in_size * count);
        batch_output_pol.resize(out_pol_size * count);
        batch_output_val.resize(out_val_size * count);
        auto index = size_t{0};
        for (auto& x : inputs) {
            std::unique_lock<std::mutex> lk(x->mutex);
            std::copy(
                begin(x->in),
                end(x->in),
                begin(batch_input) + in_size * index
            );
            index++;
        }
        if (!m_draining.load()) {
            // run the NN evaluation
            m_backend[gnum]->forward(
                batch_input,
                batch_output_pol,
                batch_output_val,
                static_cast<int>(tid),
                static_cast<int>(count)
            );
        } else {
            for (size_t i = 0; i < index; i++) {
                batch_output_pol[out_pol_size * i] = -1.0f;
            }
        }
        // Get output and copy back
        index = 0;
        for (auto& x : inputs) {
            std::copy(
                begin(batch_output_pol) + out_pol_size * index,
                begin(batch_output_pol) + out_pol_size * (index + 1),
                begin(x->out_p)
            );
            std::copy(
                begin(batch_output_val) + out_val_size * index,
                begin(batch_output_val) + out_val_size * (index + 1),
                begin(x->out_v)
            );
            x->cv.notify_all();
            index++;
        }
    }
}

template <typename net_t>
void GPUScheduler<net_t>::drain()
{
    // When signaled to drain requests, this method picks up all pending
    // requests and wakes them up.  Throws exception once the woken up request
    // sees m_draining.
    m_draining.exchange(true);
    m_cv.notify_all();
}

template <typename net_t>
void GPUScheduler<net_t>::resume()
{
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.clear();
    }
    // UCTNode::think() should wait for all child threads to complete before resuming.
    m_draining.exchange(false);
}

template class GPUScheduler<float>;
template class GPUScheduler<half_float::half>;
#endif
