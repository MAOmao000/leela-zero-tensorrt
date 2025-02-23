/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
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

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "GTP.h"
#include "GameState.h"
#include "NNCache.h"
#include "Network.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

static void license_blurb() {
    printf(
        "Leela Zero 0.17  Copyright (C) 2017-2019  Gian-Carlo Pascutto and contributors\n"
        "%s %s.%s  Copyright (C) 2024  MAOmao000 \n"
        "This program comes with ABSOLUTELY NO WARRANTY.\n"
        "This is free software, and you are welcome to redistribute it\n"
        "under certain conditions; see the COPYING file for details.\n\n",
        PROGRAM_NAME, PROGRAM_VERSION_MAJOR, PROGRAM_VERSION_MINOR);
}

static void calculate_thread_count_cpu(
    boost::program_options::variables_map& vm) {
    // If we are CPU-based, there is no point using more than the number of CPUs.
    auto cfg_max_threads = std::min(SMP::get_num_cpus(), size_t{ MAX_CPUS });
    cfg_max_threads = std::max(cfg_max_threads, size_t{1});

    if (vm["threads"].as<unsigned int>() > 0) {
        auto num_threads = vm["threads"].as<unsigned int>();
        if (num_threads > cfg_max_threads) {
            myprintf("Clamping threads to maximum = %d\n", cfg_max_threads);
            num_threads = static_cast<unsigned int>(cfg_max_threads);
        }
        cfg_num_threads = num_threads;
    } else {
        cfg_num_threads = cfg_max_threads;
    }
}

static void calculate_thread_count_gpu(
    boost::program_options::variables_map& vm) {
    auto cfg_max_threads = size_t{MAX_CPUS};

    // Default thread count : GPU case
    // 1) if no args are given, use batch size of 5 and thread count of (batch size) * (number of gpus) * 2
    // 2) if number of threads are given, use batch size of (thread count) / (number of gpus) / 2
    // 3) if number of batches are given, use thread count of (batch size) * (number of gpus) * 2
    auto gpu_count = cfg_gpus.size();
    if (gpu_count == 0) {
        // size of zero if autodetect GPU : default to 1
        gpu_count = 1;
    }

    if (vm["threads"].as<unsigned int>() > 0) {
        auto num_threads = vm["threads"].as<unsigned int>();
        if (num_threads > cfg_max_threads) {
            myprintf("Clamping threads to maximum = %d\n", cfg_max_threads);
            num_threads = static_cast<unsigned int>(cfg_max_threads);
        }
        cfg_num_threads = num_threads;

        if (vm["batchsize"].as<unsigned int>() > 0) {
            cfg_batch_size = vm["batchsize"].as<unsigned int>();
        } else {
            cfg_batch_size =
                (cfg_num_threads + (gpu_count * 2) - 1) / (gpu_count * 2);
            // no idea why somebody wants to use threads less than the number of GPUs
            // but should at least prevent crashing
            if (cfg_batch_size == 0) {
                cfg_batch_size = 1;
            } else if (cfg_batch_size > cfg_num_threads) {
                	cfg_batch_size = std::min(cfg_num_threads / 2, size_t{1});
            }
        }
    } else {
        if (vm["batchsize"].as<unsigned int>() > 0) {
            cfg_batch_size = vm["batchsize"].as<unsigned int>();
        } else {
            calculate_thread_count_cpu(vm);
            cfg_batch_size = cfg_num_threads * 5 / 6 / 2;
            if (cfg_batch_size == 0) {
                cfg_batch_size = 1;
            }
        }
        cfg_num_threads =
            std::min(cfg_max_threads, cfg_batch_size * gpu_count * 2);
    }
    if (cfg_num_threads < cfg_batch_size) {
        printf(
            "Number of threads = %zd must be no smaller than batch size = %zd\n",
            cfg_num_threads, cfg_batch_size);
        exit(EXIT_FAILURE);
    }
}

static void parse_commandline(const int argc, const char* const argv[]) {
    namespace po = boost::program_options;
    // Declare the supported options.
    po::options_description gen_desc("Generic options");
    gen_desc.add_options()
        ("help,h", "Show commandline options.")
        ("gtp,g", "Enable GTP mode.")
        ("threads,t", po::value<unsigned int>()->default_value(0),
                      "Number of threads to use. Select 0 to let leela-zero pick a reasonable default.")
        ("playouts,p", po::value<int>(),
                       "Weaken engine by limiting the number of playouts. "
                       "Requires --noponder.")
        ("visits,v", po::value<int>(),
                     "Weaken engine by limiting the number of visits.")
        ("lagbuffer,b", po::value<int>()->default_value(cfg_lagbuffer_cs),
                        "Safety margin for time usage in centiseconds.")
        ("resignpct,r", po::value<int>()->default_value(cfg_resignpct),
                        "Resign when winrate is less than x%.\n"
                        "-1 uses 5% but scales for handicap.")
        ("weights,w", po::value<std::string>()->default_value(cfg_weightsfile),
                      "File with network weights.")
        ("logfile,l", po::value<std::string>(),
                      "File to log input/output to.")
        ("quiet,q", "Disable all diagnostic output.")
        ("timemanage", po::value<std::string>()->default_value("auto"),
                       "[auto|on|off|fast|no_pruning] Enable time management features.\n"
                       "auto = no_pruning when using -n, otherwise on.\n"
                       "on = Cut off search when the best move can't change"
                       ", but use full time if moving faster doesn't save time.\n"
                       "fast = Same as on but always plays faster.\n"
                       "no_pruning = For self play training use.\n")
        ("noponder", "Disable thinking on opponent's time.")
        ("benchmark", "Test network and exit. Default args:\n-v3200 --noponder "
                      "-m0 -t1 -s1.")
        ("trt-cache", po::value<std::string>()->default_value("plan"),
                      "Which to use: plan cache or timing cache? (plan/timing)")
        ("ladder_chase", po::value<std::string>()->default_value("root"),
                      "Ladder chase check timing. (every/root/playout)")
        ("ladder_defense", po::value<int>()->default_value(cfg_ladder_defense),
                      "Ladder defense check minimum depth.")
        ("ladder_offense", po::value<int>()->default_value(cfg_ladder_offense),
                      "Ladder offense check minimum depth.")
        ("defense_stones", po::value<int>()->default_value(cfg_defense_stones),
                      "Ladder defense check minimum stones.")
        ("offense_stones", po::value<int>()->default_value(cfg_offense_stones),
                      "Ladder offense check minimum stones.")
        ("ladder_depth_defense", po::value<int>()->default_value(cfg_ladder_depth_defense),
                      "Ladder defense check maximum depth.")
        ("ladder_depth_offense", po::value<int>()->default_value(cfg_ladder_depth_offense),
                      "Ladder offense check maximum depth.")
        ("ladder_check_nodes", po::value<int>()->default_value(cfg_ladder_check_nodes),
                      "Number of nodes to check ladder.")
        ("ladder_penalty_winrate", po::value<float>()->default_value(cfg_ladder_penalty_winrate),
                      "The rate at which the ladder reduces the winning rate of the board.")
        ("chase_penalty_policy", po::value<float>()->default_value(cfg_chase_penalty_policy),
                      "The rate at which to reduce the policy if the ladder is tracked incorrectly.")
        ("chase_penalty_value", po::value<double>()->default_value(cfg_chase_penalty_value),
                      "The rate at which to reduce the value if the ladder is tracked incorrectly.")
      ;
    po::options_description gpu_desc("TensorRT device options");
    gpu_desc.add_options()
        ("gpu", po::value<std::vector<int>>(),
                "ID of the TensorRT device(s) to use (disables autodetection).")
        ("batchsize", po::value<unsigned int>()->default_value(0),
                      "Max batch size.  Select 0 to let leela-zero pick a reasonable default.")
        ("builder_opt_level", po::value<int>()->default_value(cfg_builder_opt_level),
                      "Builder optimization level.")
        ("unuse_drain_resume", "Disable drain and formula.")
        ("precision", po::value<std::string>(),
                      "Floating-point precision (single/half/auto).\n"
                      "Default is to auto which automatically determines which one to use.")
        ;
    po::options_description selfplay_desc("Self-play options");
    selfplay_desc.add_options()
        ("noise,n", "Enable policy network randomization.")
        ("seed,s", po::value<std::uint64_t>(),
                   "Random number generation seed.")
        ("dumbpass,d", "Don't use heuristics for smarter passing.")
        ("randomcnt,m", po::value<int>()->default_value(cfg_random_cnt),
                        "Play more randomly the first x moves.")
        ("randomvisits", po::value<int>()->default_value(cfg_random_min_visits),
                         "Don't play random moves if they have <= x visits.")
        ("randomtemp", po::value<float>()->default_value(cfg_random_temp),
                       "Temperature to use for random move selection.");
    po::options_description tuner_desc("Tuning options");
    tuner_desc.add_options()
        ("puct", po::value<float>())
        ("logpuct", po::value<float>())
        ("logconst", po::value<float>())
        ("dynamic_k_factor", po::value<float>())
        ("dynamic_k_base", po::value<float>())
        ("puct_stdev_scale", po::value<float>())
        ("puct_stdev_prior", po::value<float>())
        ("softmax_temp", po::value<float>())
        ("fpu_reduction", po::value<float>())
        ("ci_alpha", po::value<float>())
        ("z_entries", po::value<int>())
        ("lcb_visits_ratio", po::value<float>())
        ("unuse_stdev_uct", "Disable sample variance in UCT formula.");
    // These won't be shown, we use them to catch incorrect usage of the
    // command line.
    po::options_description ignore("Ignored options");
    po::options_description h_desc("Hidden options");
    h_desc.add_options()
        ("arguments", po::value<std::vector<std::string>>());
    po::options_description visible;
    visible
        .add(gen_desc)
        .add(gpu_desc)
        .add(selfplay_desc)
        .add(tuner_desc);
    // Parse both the above, we will check if any of the latter are present.
    po::options_description all;
    all.add(visible).add(ignore).add(h_desc);
    po::positional_options_description p_desc;
    p_desc.add("arguments", -1);
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                      .options(all)
                      .positional(p_desc)
                      .run(),
                  vm);
        po::notify(vm);
    } catch (const boost::program_options::error& e) {
        printf("ERROR: %s\n", e.what());
        license_blurb();
        std::cout << visible << std::endl;
        exit(EXIT_FAILURE);
    }

    // Handle commandline options
    if (vm.count("help") || vm.count("arguments")) {
        auto ev = EXIT_SUCCESS;
        // The user specified an argument. We don't accept any, so explain
        // our usage.
        if (vm.count("arguments")) {
            for (auto& arg : vm["arguments"].as<std::vector<std::string>>()) {
                std::cout << "Unrecognized argument: " << arg << std::endl;
            }
            ev = EXIT_FAILURE;
        }
        license_blurb();
        std::cout << visible << std::endl;
        exit(ev);
    }

    if (vm.count("quiet")) {
        cfg_quiet = true;
    }

    if (vm.count("benchmark")) {
        cfg_quiet = true; // Set this early to avoid unnecessary output.
    }

    if (vm.count("puct")) {
        cfg_puct = vm["puct"].as<float>();
    }
    if (vm.count("logpuct")) {
        cfg_logpuct = vm["logpuct"].as<float>();
    }
    if (vm.count("logconst")) {
        cfg_logconst = vm["logconst"].as<float>();
    }
    if (vm.count("dynamic_k_factor")) {
        cfg_dynamic_k_factor = vm["dynamic_k_factor"].as<float>();
    }
    if (vm.count("dynamic_k_base")) {
        cfg_dynamic_k_base = vm["dynamic_k_base"].as<float>();
    }
    if (vm.count("puct_stdev_scale")) {
        cfg_stdev_scale = vm["puct_stdev_scale"].as<float>();
    }
    if (vm.count("puct_stdev_prior")) {
        cfg_stdev_prior = vm["puct_stdev_prior"].as<float>();
    }
    if (vm.count("softmax_temp")) {
        cfg_softmax_temp = vm["softmax_temp"].as<float>();
    }
    if (vm.count("fpu_reduction")) {
        cfg_fpu_reduction = vm["fpu_reduction"].as<float>();
    }
    if (vm.count("ci_alpha")) {
        cfg_ci_alpha = vm["ci_alpha"].as<float>();
    }
    if (vm.count("z_entries")) {
        cfg_z_entries = vm["z_entries"].as<int>();
    }
    if (vm.count("lcb_visits_ratio")) {
        cfg_lcb_min_visit_ratio = vm["lcb_visits_ratio"].as<float>();
    }
    if (vm.count("unuse_stdev_uct")) {
        cfg_use_stdev_uct = false;
    }

    if (vm.count("logfile")) {
        cfg_logfile = vm["logfile"].as<std::string>();
        myprintf("Logging to %s.\n", cfg_logfile.c_str());
        cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
    }

    cfg_weightsfile = vm["weights"].as<std::string>();
    if (vm["weights"].defaulted()
        && !boost::filesystem::exists(cfg_weightsfile)) {
        printf("A network weights file is required to use the program.\n");
        printf("By default, Leela Zero looks for it in %s.\n",
               cfg_weightsfile.c_str());
        exit(EXIT_FAILURE);
    }

    if (vm.count("gtp")) {
        cfg_gtp_mode = true;
    }

    if (vm.count("gpu")) {
        cfg_gpus = vm["gpu"].as<std::vector<int>>();
    }

    if (vm.count("builder_opt_level")) {
        cfg_builder_opt_level = vm["builder_opt_level"].as<int>();
    }

    if (vm.count("unuse_drain_resume")) {
        cfg_use_drain_resume = false;
    }

    auto trt_cache = vm["trt-cache"].as<std::string>();
    if ("plan" == trt_cache) {
        cfg_cache_plan = true;
    } else if ("timing" == trt_cache) {
        cfg_cache_plan = false;
    } else {
        printf("Unexpected option for --trt-cache, expecting plan/timing.\n");
        exit(EXIT_FAILURE);
    }
    calculate_thread_count_gpu(vm);
    myprintf("Using TensorRT batch size of %d\n", cfg_batch_size);
    myprintf("Using %d thread(s).\n", cfg_num_threads);

    if (vm.count("precision")) {
        auto precision = vm["precision"].as<std::string>();
        if ("single" == precision) {
            cfg_precision = precision_t::SINGLE;
        } else if ("half" == precision) {
            cfg_precision = precision_t::HALF;
        } else if ("auto" == precision) {
            // Auto precision is not supported for full tuner cases.
            cfg_precision = precision_t::AUTO;
        } else {
            printf("Unexpected option for --precision, expecting single/half/auto\n");
            exit(EXIT_FAILURE);
        }
    }

    if (vm.count("seed")) {
        cfg_rng_seed = vm["seed"].as<std::uint64_t>();
        if (cfg_num_threads > 1) {
            myprintf("Seed specified but multiple threads enabled.\n");
            myprintf("Games will likely not be reproducible.\n");
        }
    }
    myprintf("RNG seed: %llu\n", cfg_rng_seed);

    if (vm.count("noponder")) {
        cfg_allow_pondering = false;
    }

    if (vm.count("noise")) {
        cfg_noise = true;
    }

    if (vm.count("dumbpass")) {
        cfg_dumbpass = true;
    }

    if (vm.count("playouts")) {
        cfg_max_playouts = vm["playouts"].as<int>();
        if (!vm.count("noponder")) {
            printf("Nonsensical options: Playouts are restricted but "
                   "thinking on the opponent's time is still allowed. "
                   "Add --noponder if you want a weakened engine.\n");
            exit(EXIT_FAILURE);
        }

        // 0 may be specified to mean "no limit"
        if (cfg_max_playouts == 0) {
            cfg_max_playouts = UCTSearch::UNLIMITED_PLAYOUTS;
        }
    }

    if (vm.count("visits")) {
        cfg_max_visits = vm["visits"].as<int>();

        // 0 may be specified to mean "no limit"
        if (cfg_max_visits == 0) {
            cfg_max_visits = UCTSearch::UNLIMITED_PLAYOUTS;
        }
    }

    if (vm.count("resignpct")) {
        cfg_resignpct = vm["resignpct"].as<int>();
    }

    if (vm.count("randomcnt")) {
        cfg_random_cnt = vm["randomcnt"].as<int>();
    }

    if (vm.count("randomvisits")) {
        cfg_random_min_visits = vm["randomvisits"].as<int>();
    }

    if (vm.count("randomtemp")) {
        cfg_random_temp = vm["randomtemp"].as<float>();
    }

    if (vm.count("timemanage")) {
        auto tm = vm["timemanage"].as<std::string>();
        if (tm == "auto") {
            cfg_timemanage = TimeManagement::AUTO;
        } else if (tm == "on") {
            cfg_timemanage = TimeManagement::ON;
        } else if (tm == "off") {
            cfg_timemanage = TimeManagement::OFF;
        } else if (tm == "fast") {
            cfg_timemanage = TimeManagement::FAST;
        } else if (tm == "no_pruning") {
            cfg_timemanage = TimeManagement::NO_PRUNING;
        } else {
            printf("Invalid timemanage value.\n");
            exit(EXIT_FAILURE);
        }
    }
    if (cfg_timemanage == TimeManagement::AUTO) {
        cfg_timemanage =
            cfg_noise ? TimeManagement::NO_PRUNING : TimeManagement::ON;
    }

    if (vm.count("lagbuffer")) {
        int lagbuffer = vm["lagbuffer"].as<int>();
        if (lagbuffer != cfg_lagbuffer_cs) {
            myprintf("Using per-move time margin of %.2fs.\n",
                     lagbuffer / 100.0f);
            cfg_lagbuffer_cs = lagbuffer;
        }
    }
    if (vm.count("benchmark")) {
        // These must be set later to override default arguments.
        cfg_allow_pondering = false;
        cfg_benchmark = true;
        cfg_noise = false; // Not much of a benchmark if random was used.
        cfg_random_cnt = 0;
        cfg_rng_seed = 1;
        cfg_timemanage = TimeManagement::OFF; // Reliable number of playouts.

        if (!vm.count("playouts") && !vm.count("visits")) {
            cfg_max_visits = 3200; // Default to self-play and match values.
        }
    }

    // Do not lower the expected eval for root moves that are likely not
    // the best if we have introduced noise there exactly to explore more.
    cfg_fpu_root_reduction = cfg_noise ? 0.0f : cfg_fpu_reduction;

    if (vm.count("ladder_chase")) {
        auto ladder_chase = vm["ladder_chase"].as<std::string>();
        if (ladder_chase == "every") {
            cfg_ladder_chase = chase_t::EVERY;
        } else if (ladder_chase == "root") {
            cfg_ladder_chase = chase_t::ROOT;
        } else if (ladder_chase == "playout") {
            cfg_ladder_chase = chase_t::PLAYOUT;
        } else {
            printf("Invalid ladder_chase value.\n");
            exit(EXIT_FAILURE);
        }
    }

    if (vm.count("ladder_defense")) {
        cfg_ladder_defense = vm["ladder_defense"].as<int>();
    }

    if (vm.count("ladder_offense")) {
        cfg_ladder_offense = vm["ladder_offense"].as<int>();
    }

    if (vm.count("defense_stones")) {
        cfg_defense_stones = vm["defense_stones"].as<int>();
    }

    if (vm.count("offense_stones")) {
        cfg_offense_stones = vm["offense_stones"].as<int>();
    }

    if (vm.count("ladder_depth_defense")) {
        cfg_ladder_depth_defense = vm["ladder_depth_defense"].as<int>();
    }

    if (vm.count("ladder_depth_offense")) {
        cfg_ladder_depth_offense = vm["ladder_depth_offense"].as<int>();
    }

    if (vm.count("ladder_check_nodes")) {
        cfg_ladder_check_nodes = vm["ladder_check_nodes"].as<int>();
    }

    if (vm.count("ladder_penalty_winrate")) {
        cfg_ladder_penalty_winrate = vm["ladder_penalty_winrate"].as<float>();
    }

    if (vm.count("chase_penalty_policy")) {
        cfg_chase_penalty_policy = vm["chase_penalty_policy"].as<float>();
    }

    if (vm.count("chase_penalty_value")) {
        cfg_chase_penalty_value = vm["chase_penalty_value"].as<double>();
    }

    auto out = std::stringstream{};
    for (auto i = 1; i < argc; i++) {
        out << " " << argv[i];
    }
    if (!vm.count("seed")) {
        out << " --seed " << cfg_rng_seed;
    }
    cfg_options_str = out.str();
}

static void initialize_network() {
    auto network = std::make_unique<Network>();
    auto playouts = std::min(cfg_max_playouts, cfg_max_visits);
    network->initialize(playouts, cfg_weightsfile);

    GTP::initialize(std::move(network));
}

// Setup global objects after command line has been parsed
void init_global_objects() {
    thread_pool.initialize(cfg_num_threads);

    // Use deterministic random numbers for hashing
    auto rng = std::make_unique<Random>(5489);
    Zobrist::init_zobrist(*rng);

    // Initialize the main thread RNG.
    // Doing this here avoids mixing in the thread_id, which
    // improves reproducibility across platforms.
    Random::get_Rng().seedrandom(cfg_rng_seed);

    Utils::create_z_table();

    initialize_network();
}

void benchmark(GameState& game) {
    game.set_timecontrol(0, 1, 0, 0); // Set infinite time.
    game.play_textmove("b", "r16");
    game.play_textmove("w", "d4");
    game.play_textmove("b", "c3");

    auto search = std::make_unique<UCTSearch>(game, *GTP::s_network);
    game.set_to_move(FastBoard::WHITE);
    search->think(FastBoard::WHITE);
}

int main(int argc, char* argv[]) {
    // Set up engine parameters
    GTP::setup_default_parameters();
    parse_commandline(argc, argv);

    // Disable IO buffering as much as possible
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    std::cin.setf(std::ios::unitbuf);

    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
#ifndef _WIN32
    setbuf(stdin, nullptr);
#endif

    if (!cfg_gtp_mode && !cfg_benchmark) {
        license_blurb();
    }

    init_global_objects();

    auto maingame = std::make_unique<GameState>();

    /* set board limits */
    maingame->init_game(BOARD_SIZE, KOMI);

    if (cfg_benchmark) {
        cfg_quiet = false;
        benchmark(*maingame);
        return 0;
    }

    for (;;) {
        if (!cfg_gtp_mode) {
            maingame->display_state();
            std::cout << "Leela: ";
        }

        auto input = std::string{};
        if (std::getline(std::cin, input)) {
            Utils::log_input(input);
            GTP::execute(*maingame, input);
        } else {
            // eof or other error
            std::cout << std::endl;
            break;
        }

        // Force a flush of the logfile
        if (cfg_logfile_handle) {
            fclose(cfg_logfile_handle);
            cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
        }
    }

    return 0;
}
