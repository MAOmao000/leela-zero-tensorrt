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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "config.h"

#include <array>
#include <deque>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FastState.h"
#include "NNCache.h"
#include "ForwardPipe.h"
#include "GameState.h"

// Winograd filter transformation changes 3x3 filters to M + 3 - 1
constexpr auto FILTER_SIZE = 3;

class Network {
    using ForwardPipeWeights = ForwardPipe::ForwardPipeWeights;

public:
    static constexpr auto NUM_SYMMETRIES = 8;
    static constexpr auto IDENTITY_SYMMETRY = 0;
    enum Ensemble {
        DIRECT, RANDOM_SYMMETRY, AVERAGE
    };
    using PolicyVertexPair = std::pair<float, int>;
    using Netresult = NNCache::Netresult;

    ~Network() = default;

    bool get_output(const GameState* state, const Ensemble ensemble,
                    Network::Netresult& result,
                    const int symmetry = -1, const bool read_cache = true,
                    const bool write_cache = true);

    static constexpr auto INPUT_MOVES = 8;
    static constexpr auto INPUT_CHANNELS = 2 * INPUT_MOVES + 2;
    static constexpr auto OUTPUTS_POLICY = 2;
    static constexpr auto OUTPUTS_VALUE = 1;
    static constexpr auto VALUE_LAYER = 256;

    void initialize(int playouts, const std::string& weightsfile);

    void benchmark(const GameState* state, int iterations = 1600);
    static void show_heatmap(const FastState* state, const Netresult& netres,
                             bool topmoves);

    static std::vector<float> gather_features(const GameState* state,
                                              int symmetry);
    static std::pair<int, int> get_symmetry(const std::pair<int, int>& vertex,
                                            int symmetry,
                                            int board_size = BOARD_SIZE);

    size_t get_estimated_size();
    size_t get_estimated_cache_size();
    void nncache_resize(int max_count);
    void nncache_clear();

    void drain_evals();

    // Flag the network to be open for business.
    void resume_evals();

    NetworkType get_network_type() {
        return m_net_type;
    }

private:
    std::pair<int, int> load_v1_network(std::istream& wtfile);
    std::pair<int, int> load_network_file(const std::string& filename);

    bool get_output_internal(const GameState* state,
                             int symmetry,
                             Network::Netresult& result);
    void ladder_update(const GameState* state, Network::Netresult& result);
    static void fill_input_plane_pair(const FullBoard& board,
                                      std::vector<float>::iterator black,
                                      std::vector<float>::iterator white,
                                      int symmetry);
    bool probe_cache(const GameState* state, Network::Netresult& result);
    std::unique_ptr<ForwardPipe>&& init_net(
        int channels, std::unique_ptr<ForwardPipe>&& pipe);
    void select_precision(int channels);
    std::unique_ptr<ForwardPipe> m_forward;

    NNCache m_nncache;

    size_t estimated_size{0};

    // Residual tower
    std::shared_ptr<ForwardPipeWeights> m_fwd_weights;

    // Policy head
    std::array<float, OUTPUTS_POLICY> m_bn_pol_w1{};
    std::array<float, OUTPUTS_POLICY> m_bn_pol_w2{};

    std::array<float, OUTPUTS_POLICY * NUM_INTERSECTIONS * POTENTIAL_MOVES>
        m_ip_pol_w{};
    std::array<float, POTENTIAL_MOVES> m_ip_pol_b{};

    // Value head
    std::array<float, OUTPUTS_VALUE> m_bn_val_w1{};
    std::array<float, OUTPUTS_VALUE> m_bn_val_w2{};

    std::array<float, OUTPUTS_VALUE * NUM_INTERSECTIONS * VALUE_LAYER>
        m_ip1_val_w{};
    std::array<float, VALUE_LAYER> m_ip1_val_b{};

    std::array<float, VALUE_LAYER> m_ip2_val_w{};
    std::array<float, 1> m_ip2_val_b{};
    bool m_value_head_not_stm{};

    std::string m_model_hash{""};

    NetworkType m_net_type{ NetworkType::LEELA_ZERO };
};
#endif
