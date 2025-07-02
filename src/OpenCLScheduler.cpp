/*
    This file is part of Leela Zero.
    Copyright (C) 2018-2019 Junhee Yoo and contributors

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

#if defined(USE_OPENCL)

#include "GTP.h"
#include "Network.h"
#include "OpenCLScheduler.h"
#include "Random.h"
#include "Utils.h"
#include "CPUPipe.h"

using Utils::ceilMultiple;
using Utils::myprintf;

class from_float {
public:
    from_float(const std::vector<float>& f) : m_f(f) {}

    operator const std::vector<float> &() {
        return m_f;
    }

    operator std::vector<half_float::half>() {
        auto ret = std::vector<half_float::half>(m_f.size());
        std::copy(cbegin(m_f), cend(m_f), begin(ret));
        return ret;
    }

private:
    const std::vector<float>& m_f;
};

template <typename T>
static std::vector<T> zeropad_U(const std::vector<float>& U, const int outputs,
                                const int channels, const int outputs_pad,
                                const int channels_pad) {
    // Fill with zeroes
    auto Upad = std::vector<T>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
        for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
            for (auto c = 0; c < channels; c++) {
                for (auto o = 0; o < outputs; o++) {
                    Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad) + c * outputs_pad
                         + o] =
                        U[xi * (WINOGRAD_ALPHA * outputs * channels)
                          + nu * (outputs * channels) + c * outputs + o];
                }
            }
        }
    }

    return Upad;
}

template <typename net_t>
OpenCLScheduler<net_t>::OpenCLScheduler() {
    // multi-gpu?
    auto gpus = cfg_gpus;

    // An empty GPU list from the command line represents autodetect.
    // Put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }

    auto silent{false};

    for (auto gpu : gpus) {
        auto opencl = std::make_unique<OpenCL<net_t>>(gpu, silent);
        auto net = std::make_unique<OpenCL_Network<net_t>>(*opencl);
        m_opencl.push_back(std::move(opencl));
        m_networks.push_back(std::move(net));

        // Starting next GPU, let's not dump full list of GPUs.
        silent = true;
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::initialize(
    const int channels, const NetworkType net_type, const std::string &model_hash) {

    (void) model_hash;

    m_net_type = net_type;
    // Launch the worker threads.  Minimum 1 worker per GPU, but use enough
    // threads so that we can at least concurrently schedule something to the
    // GPU.
    auto num_worker_threads =
        cfg_num_threads / cfg_batch_size / (m_opencl.size() + 1) + 1;
    auto gnum = 0;
    for (auto& opencl : m_opencl) {
        opencl->initialize(channels, cfg_batch_size, net_type);

        for (auto i = unsigned{0}; i < num_worker_threads; i++) {
            auto t =
                std::thread(&OpenCLScheduler<net_t>::batch_worker, this, gnum);
            m_worker_threads.push_back(std::move(t));
        }
        gnum++;
    }

    // Exit immediately after tuning.  We should exit here because we skipped
    // initializing rest of the kernels due to some NVIDIA drivers crashing.
    if (cfg_tune_only) {
        exit(EXIT_SUCCESS);
    }
}

template <typename net_t>
OpenCLScheduler<net_t>::~OpenCLScheduler() {
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_running = false;
    }
    m_cv.notify_all();
    for (auto& x : m_worker_threads) {
        x.join();
    }
}

template <typename net_t>
bool OpenCLScheduler<net_t>::needs_autodetect() {
    for (auto& opencl : m_opencl) {
        // If any card has no native fp16 compute, we'll have to benchmark.
        if (!opencl->has_fp16_compute() && !opencl->has_tensor_cores()) {
            return true;
        }
    }
    return false;
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_input_convolution(
    const unsigned int filter_size, const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& means,
    const std::vector<float>& variances) {

    for (const auto& opencl_net : m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto kwg = tuners[2];
        const auto vwm = tuners[3];

        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto k_ceil = ceilMultiple(ceilMultiple(channels, kwg), vwm);

        const auto Upad =
            zeropad_U<net_t>(weights, outputs, channels, m_ceil, k_ceil);
        opencl_net->push_input_convolution(filter_size, channels, outputs, Upad,
                                           from_float(means),
                                           from_float(variances));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_residual(
    const unsigned int filter_size, const unsigned int channels,
    const unsigned int outputs,
    const std::vector<float>& weights_1,
    const std::vector<float>& means_1,
    const std::vector<float>& variances_1,
    const std::vector<float>& weights_2,
    const std::vector<float>& means_2,
    const std::vector<float>& variances_2) {
    for (const auto& opencl_net : m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto vwm = tuners[3];

        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto Upad1 =
            zeropad_U<net_t>(weights_1, outputs, outputs, m_ceil, m_ceil);
        const auto Upad2 =
            zeropad_U<net_t>(weights_2, outputs, outputs, m_ceil, m_ceil);
        opencl_net->push_residual(filter_size, channels, outputs,
                                  Upad1, from_float(means_1),
                                  from_float(variances_1),
                                  Upad2, from_float(means_2),
                                  from_float(variances_2));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_residual_se(
    const unsigned int filter_size,
    const unsigned int channels,
    const unsigned int outputs,
    const size_t weight_index,
    const std::shared_ptr<const ForwardPipe::ForwardPipeWeights> weights) {

    for (const auto& opencl_net : m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();
        const auto mwg = tuners[0];
        const auto vwm = tuners[3];
        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto Upad1 = zeropad_U<net_t>(
            weights->m_conv_weights[weight_index],
            outputs,
            outputs,
            m_ceil,
            m_ceil
        );
        const auto Upad2 = zeropad_U<net_t>(
            weights->m_conv_weights[weight_index + 1],
            outputs,
            outputs,
            m_ceil,
            m_ceil
        );
        opencl_net->push_residual_se(
            filter_size,
            channels,
            outputs,
            Upad1,
            from_float(weights->m_batchnorm_means[weight_index]),
            from_float(weights->m_batchnorm_stddevs[weight_index]),
            Upad2,
            from_float(weights->m_batchnorm_means[weight_index + 1]),
            from_float(weights->m_batchnorm_stddevs[weight_index + 1]),
            from_float(weights->m_se_weights[weight_index - 1]),
            from_float(weights->m_se_biases[weight_index - 1]),
            from_float(weights->m_se_weights[weight_index]),
            from_float(weights->m_se_biases[weight_index])
        );
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_convolve(const unsigned int filter_size,
                                           const unsigned int channels,
                                           const unsigned int outputs,
                                           const std::vector<float>& weights) {
    for (const auto& opencl_net : m_networks) {
        opencl_net->push_convolve(filter_size, channels, outputs,
                                  from_float(weights));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_weights(
    const unsigned int filter_size, const unsigned int channels,
    const unsigned int outputs,
    std::shared_ptr<const ForwardPipeWeights> weights) {

    auto weight_index = size_t{0};

    // Winograd filter transformation changes filter size to 4x4
    push_input_convolution(filter_size, channels, outputs,
                           weights->m_conv_weights[weight_index],
                           weights->m_batchnorm_means[weight_index],
                           weights->m_batchnorm_stddevs[weight_index]);
    weight_index++;

    if (m_net_type == NetworkType::LEELA_ZERO) {
        // residual blocks : except the first entry,
        // the second ~ last entry is all on residual topwer
        for (auto i = size_t{0}; i < weights->m_conv_weights.size() / 2; i++) {
            push_residual(filter_size, outputs, outputs,
                          weights->m_conv_weights[weight_index],
                          weights->m_batchnorm_means[weight_index],
                          weights->m_batchnorm_stddevs[weight_index],
                          weights->m_conv_weights[weight_index + 1],
                          weights->m_batchnorm_means[weight_index + 1],
                          weights->m_batchnorm_stddevs[weight_index + 1]);
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
    push_convolve(1, outputs, Network::OUTPUTS_POLICY, weights->m_conv_pol_w);
    push_convolve(1, outputs, Network::OUTPUTS_VALUE, weights->m_conv_val_w);

    m_bn_pol_w1 = weights->m_bn_pol_w1;
    m_bn_pol_w2 = weights->m_bn_pol_w2;
    m_ip_pol_w = weights->m_ip_pol_w;
    m_ip_pol_b = weights->m_ip_pol_b;
    m_bn_val_w1 = weights->m_bn_val_w1;
    m_bn_val_w2 = weights->m_bn_val_w2;
    m_ip1_val_w = weights->m_ip1_val_w;
    m_ip1_val_b = weights->m_ip1_val_b;
    m_ip2_val_w = weights->m_ip2_val_w;
    m_ip2_val_b = weights->m_ip2_val_b;
}

template <typename net_t>
bool OpenCLScheduler<net_t>::forward(const std::vector<float>& input,
                                     std::vector<float>& output_pol,
                                     std::vector<float>& output_val,
                                     const bool full_batch) {
    if (m_draining.load()) {
        return false;
    }
    std::vector<float> policy_data(Network::OUTPUTS_POLICY * NUM_INTERSECTIONS);
    std::vector<float> value_data(Network::OUTPUTS_VALUE * NUM_INTERSECTIONS);
    auto entry =
        std::make_shared<ForwardQueueEntry>(input, policy_data, value_data, full_batch);
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

    if (policy_data[0] == -1.0f) {
        return false;
    }
    // Get the moves
    CPUPipe::batchnorm<NUM_INTERSECTIONS>(Network::OUTPUTS_POLICY, policy_data,
                                          m_bn_pol_w1.data(),
                                          m_bn_pol_w2.data());
    const auto policy_out =
        CPUPipe::innerproduct_pub<Network::OUTPUTS_POLICY * NUM_INTERSECTIONS, POTENTIAL_MOVES, false>
            (policy_data, m_ip_pol_w, m_ip_pol_b);
    output_pol = Utils::softmax(policy_out, cfg_softmax_temp);

    // Now get the value
    CPUPipe::batchnorm<NUM_INTERSECTIONS>(Network::OUTPUTS_VALUE, value_data,
                                          m_bn_val_w1.data(),
                                          m_bn_val_w2.data());
    const auto winrate_data =
        CPUPipe::innerproduct_pub<Network::OUTPUTS_VALUE * NUM_INTERSECTIONS, Network::VALUE_LAYER, true>
            (value_data, m_ip1_val_w, m_ip1_val_b);
    const auto winrate_out =
        CPUPipe::innerproduct_pub<Network::VALUE_LAYER, 1, false>
            (winrate_data, m_ip2_val_w, m_ip2_val_b);

    output_val[0] = std::tanh(winrate_out[0]);

    return true;
}

template <typename net_t>
void OpenCLScheduler<net_t>::batch_worker(const size_t gnum) {
    constexpr auto in_size = Network::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;
    constexpr auto out_pol_size =
        Network::OUTPUTS_POLICY * BOARD_SIZE * BOARD_SIZE;
    constexpr auto out_val_size =
        Network::OUTPUTS_VALUE * BOARD_SIZE * BOARD_SIZE;

    OpenCLContext context;

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
    auto batch_input = std::vector<float>();
    auto batch_output_pol = std::vector<float>();
    auto batch_output_val = std::vector<float>();
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
            std::copy(begin(x->in), end(x->in),
                      begin(batch_input) + in_size * index);
            index++;
        }
        if (!m_draining.load()) {
            // run the NN evaluation
            m_networks[gnum]->forward(batch_input, batch_output_pol,
                                      batch_output_val, context, count);
        } else {
            for (size_t i = 0; i < index; i++) {
                batch_output_pol[out_pol_size * i] = -1.0f;
            }
        }
        // Get output and copy back
        index = 0;
        for (auto& x : inputs) {
            std::copy(begin(batch_output_pol) + out_pol_size * index,
                      begin(batch_output_pol) + out_pol_size * (index + 1),
                      begin(x->out_p));
            std::copy(begin(batch_output_val) + out_val_size * index,
                      begin(batch_output_val) + out_val_size * (index + 1),
                      begin(x->out_v));
            x->cv.notify_all();
            index++;
        }
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::drain() {
    // When signaled to drain requests, this method picks up all pending
    // requests and wakes them up.  Throws exception once the woken up request
    // sees m_draining.
    m_draining.exchange(true);
    m_cv.notify_all();
}

template <typename net_t>
void OpenCLScheduler<net_t>::resume() {
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.clear();
    }
    // UCTNode::think() should wait for all child threads to complete before resuming.
    m_draining.exchange(false);
}

template class OpenCLScheduler<float>;
template class OpenCLScheduler<half_float::half>;

#endif
