#include <iostream>
#include <memory>
#include <cmath>

#include "LadderDetection.h"
#include "GTP.h"
#include "Utils.h"

using namespace std;
using namespace Utils;

#define ALIVE  1
#define DEAD   0
#define CHECKED 1
#define FLIP_COLOR(col) ((col) ^ 0x01)

static bool LadderEscape(
    int &depth,
    GameState *state,
    const int str_vtx,
    const bool escape = true,
    int escape_pos = -1);

static bool LadderChase(
    int &depth,
    GameState *state,
    const int str_vtx,
    const bool escape = true);

static bool LadderEscape(
    int &depth,
    GameState *state,
    const int str_vtx,
    const bool escape,
    int escape_pos)
{
    if (escape) {
        if (depth >= cfg_ladder_depth_defense) {
            return ALIVE;
        }
    } else {
        if (depth >= cfg_ladder_depth_offense) {
            depth = INT_MAX;
            return ALIVE;
        }
    }
    auto escape_color = state->board.get_state(str_vtx);
    if (escape_color == FastBoard::EMPTY) {
        return DEAD;
    }
    auto num_liberty = state->board.get_liberties(str_vtx);
    auto base_depth = depth;

    auto max_depth_alive = 0;
    auto min_depth_dead = INT_MAX;
    if (num_liberty >= 2) {
        return ALIVE;
    }
    // Check if can capture the stone of the surrounding opponent.
    char capture_checked[FastBoard::NUM_VERTICES] = {};
    const auto turn_color = state->board.get_to_move();
    auto newpos = str_vtx;
    auto n_vtx = 0;
    do {
        // Check whether can capture the stone at the breathing point of opponent's stones.
        for (auto d = 0; d < 4; d++) {
            n_vtx = state->board.get_state_neighbor(newpos, d);
            if (state->board.get_state(n_vtx) != FLIP_COLOR(escape_color) ||
                state->board.get_liberties(n_vtx) != 1) {
                continue;
            }
            if (capture_checked[state->board.get_parent_stone(n_vtx)]) {
                continue;
            }
            capture_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
            auto liberty_pos = state->board.get_liberty_pos(1, n_vtx);
            if (state->is_move_legal(turn_color, liberty_pos[0])) {
                state->play_move(turn_color, liberty_pos[0]);
                depth = base_depth;
                if (LadderChase(
                        ++depth,
                        state,
                        str_vtx,
                        escape
                    ) == ALIVE) {
                    if (escape) {
                        state->undo_move();
                        return ALIVE;
                    }
                    max_depth_alive = std::max(depth, max_depth_alive);
                } else {
                    min_depth_dead = std::min(depth, min_depth_dead);
                }
                state->undo_move();
            }
        }
        newpos = state->board.get_next_stone(newpos);
    } while (newpos != str_vtx);

    if (escape_pos < 0) {
        auto liberty_pos = state->board.get_liberty_pos(1, str_vtx);
        escape_pos = liberty_pos[0];
    }
    if (state->is_move_legal(turn_color, escape_pos)) {
        state->play_move(turn_color, escape_pos);
        depth = base_depth;
        if (LadderChase(
                ++depth,
                state,
                str_vtx,
                escape
            ) == ALIVE) {
            state->undo_move();
            depth = std::max(depth, max_depth_alive);
            return ALIVE;
        } else {
            state->undo_move();
            if (max_depth_alive) {
                depth = max_depth_alive;
                return ALIVE;
            }
            depth = std::min(depth, min_depth_dead);
            return DEAD;
        }
    }
    if (max_depth_alive) {
        depth = max_depth_alive;
        return ALIVE;
    } else if (min_depth_dead < INT_MAX) {
        depth = min_depth_dead;
    }
    return DEAD;
}

static bool LadderChase(
    int &depth,
    GameState *state,
    const int str_vtx,
    const bool escape)
{
    if (escape) {
        if (depth >= cfg_ladder_depth_defense) {
            return ALIVE;
        }
    } else {
        if (depth >= cfg_ladder_depth_offense) {
            depth = INT_MAX;
            return ALIVE;
        }
    }
    auto escape_color = state->board.get_state(str_vtx);
    if (escape_color == FastBoard::EMPTY) {
        return DEAD;
    }
    auto num_liberty = state->board.get_liberties(str_vtx);
    auto base_depth = depth;
    if (num_liberty >= 3) {
        return ALIVE;
    } else if (num_liberty <= 1) {
        return DEAD;
    }
    const auto turn_color = state->board.get_to_move();
    auto max_depth_alive = 0;
    auto min_depth_dead = INT_MAX;
    auto liberty_pos = state->board.get_liberty_pos(2, str_vtx);
    for (auto i = 0; i < 2; i++) {
        if (liberty_pos[i] && state->is_move_legal(turn_color, liberty_pos[i])) {
            state->play_move(turn_color, liberty_pos[i]);
            depth = base_depth;
            if (LadderEscape(
                ++depth,
                state,
                str_vtx,
                escape
                ) == DEAD) {
                if (!escape) {
                    state->undo_move();
                    return DEAD;
                }
                min_depth_dead = std::min(depth, min_depth_dead);
            } else {
                max_depth_alive = std::max(depth, max_depth_alive);
            }
            state->undo_move();
        //} else {
        // liberty_pos[i] is_suicide
        }
    }
    if (min_depth_dead < INT_MAX) {
        depth = min_depth_dead;
        return DEAD;
    }
    depth = std::max(depth, max_depth_alive);
    return ALIVE;
}

void LadderDetection(
    GameState* const state,
    int *ladder_pos,
    const std::array<float, NUM_INTERSECTIONS>& policy,
    const float ladder_min_policy,
    const int check_nodes
    )
{
    const auto turn_color = state->board.get_to_move();
    const auto opponent_color = FLIP_COLOR(turn_color);

    int check_nodes_count = 0;
    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        if (check_nodes_count >= check_nodes) {
            break;
        }
        if (policy[i] < ladder_min_policy) {
            continue;
        }
        check_nodes_count++;
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state->board.get_vertex(x, y);

        if (state->board.get_state(vertex) != FastBoard::EMPTY
            || !state->is_move_legal(turn_color, vertex)) {
            continue;
        }

        state->play_move(turn_color, vertex);
        if (cfg_ladder_defense > 0) {
            if (state->board.get_string_count(vertex) >= cfg_defense_stones
                && state->board.get_liberties(vertex) == 2) {
                auto depth = 0;
                if (LadderChase(
                    depth,
                    state,
                    vertex,
                    true
                    ) == DEAD) {
                    ladder_pos[i] = depth;
                    state->undo_move();
                    continue;
                }
            }
        }

        if (cfg_ladder_offense < 1 || cfg_ladder_chase != chase_t::EVERY) {
            state->undo_move();
            continue;
        }

        auto str_vtx = -1;
        auto liberty_vtx = -1;
        char ladder_checked[FastBoard::NUM_VERTICES] = {};
        for (auto d = 0; d < 4; d++) {
            auto n_vtx = state->board.get_state_neighbor(vertex, d);
            if (state->board.get_state(n_vtx) == opponent_color
                && !ladder_checked[state->board.get_parent_stone(n_vtx)]
                && state->board.get_liberties(n_vtx) == 1) {
                ladder_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
                auto liberty_pos = state->board.get_liberty_pos(1, n_vtx);
                if (state->is_move_legal(opponent_color, liberty_pos[0])) {
                    if (liberty_vtx != -1) {
                        liberty_vtx = -1;
                        break;
                    }
                    str_vtx = n_vtx;
                    liberty_vtx = liberty_pos[0];
                }
            }
        }
        if (liberty_vtx != -1
            && state->board.get_string_count(str_vtx) >= cfg_offense_stones) {
            auto depth = 0;
            if (LadderEscape(
                depth,
                state,
                str_vtx,
                false,
                liberty_vtx
                ) == ALIVE) {
                ladder_pos[i] = -depth;
            }
        }
        state->undo_move();
    }
}

bool IsLadderRoot(
    GameState* const state,
    const int move_vertex
    )
{
    if (cfg_ladder_chase == chase_t::EVERY
        || cfg_ladder_offense < 1
        || state->get_movenum() < NUM_INTERSECTIONS / 10
        || state->get_movenum() > NUM_INTERSECTIONS / 2
        || state->m_komove != FastBoard::NO_VERTEX) {
        return false;
    }
    const auto turn_color = state->board.get_to_move();
    const auto opponent_color = FLIP_COLOR(turn_color);

    state->play_move(turn_color, move_vertex);

    auto str_vtx = -1;
    auto liberty_vtx = -1;
    char ladder_checked[FastBoard::NUM_VERTICES] = {};
    for (auto d = 0; d < 4; d++) {
        auto n_vtx = state->board.get_state_neighbor(move_vertex, d);
        if (state->board.get_state(n_vtx) == opponent_color
            && !ladder_checked[state->board.get_parent_stone(n_vtx)]
            && state->board.get_string_count(n_vtx) >= cfg_offense_stones
            && state->board.get_liberties(n_vtx) == 1) {
            ladder_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
            auto liberty_pos = state->board.get_liberty_pos(1, n_vtx);
            if (state->is_move_legal(opponent_color, liberty_pos[0])) {
                if (liberty_vtx != -1) {
                    state->undo_move();
                    return false;
                }
                str_vtx = n_vtx;
                liberty_vtx = liberty_pos[0];
            }
        }
    }
    if (liberty_vtx != -1) {
        auto depth = 0;
        if (LadderEscape(
            depth,
            state,
            str_vtx,
            false,
            liberty_vtx
            ) == DEAD) {
            state->undo_move();
            return false;
        }
        if (depth >= cfg_ladder_offense) {
            state->undo_move();
            return true;
        }
    }
    state->undo_move();
    return false;
}
