#include <iostream>
#include <memory>
#include <cmath>

#include "LadderDetection.h"
#include "GTP.h"
#include "Utils.h"

using namespace std;
using namespace Utils;

#define ALIVE false
#define DEAD true
#define CHECKED 1
#define FLIP_COLOR(col) ((col) ^ 0x01)

static constexpr int MAX_LADDER_SEARCH_NODE_BUDGET = 25000;
static constexpr int stackSize = NUM_INTERSECTIONS * 3 / 2 + 1;

static bool LadderExploration(
    std::unique_ptr<GameState> &state, 
    const int &str_vtx,
    int &depth)
{
    int moveListStarts[stackSize];
    int moveListLens[stackSize];
    int moveListCur[stackSize];
    std::array<int, stackSize * 2> buf;

    FastBoard::vertex_t pla = state->board.get_state(str_vtx);
    FastBoard::vertex_t opp = static_cast<FastBoard::vertex_t>(FLIP_COLOR(pla));

    int stackIdx = 0;
    int searchNodeCount = 0;
    moveListCur[0] = -1;
    moveListStarts[0] = 0;
    moveListLens[0] = 0;
    bool returnValue = ALIVE;
    bool returnedFromDeeper = false;

    while (true) {
        if (stackIdx <= -1) {
            return returnValue;
        }
        if (stackIdx >= stackSize - 1) {
            returnValue = DEAD;
            returnedFromDeeper = true;
            stackIdx--;
            continue;
        }
        if (searchNodeCount >= MAX_LADDER_SEARCH_NODE_BUDGET) {
            stackIdx--;
            while (stackIdx >= 0) {
                state->undo_move();
                stackIdx--;
            }
            return DEAD;
        }
        bool isDefender = ((stackIdx % 2) == 0);

        if (moveListCur[stackIdx] == -1) {
            if (pla == FastBoard::EMPTY) {
                returnValue = DEAD;
                returnedFromDeeper = true;
                stackIdx--;
                continue;
            }
            int libs = state->board.get_liberties(str_vtx);
            if (isDefender) {
                if (libs >= 2) {
                    returnValue = ALIVE;
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }
            } else {
                if (libs <= 1) {
                    returnValue = DEAD;
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                } else if (libs >= 3) {
                    returnValue = ALIVE;
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }
            }
            int start = moveListStarts[stackIdx];
            int moveListLen = 0;
            if (isDefender) {
                // Check if can capture the stone of the surrounding opponent.
                unsigned char capture_checked[FastBoard::NUM_VERTICES] = {};
                unsigned char breath_checked[FastBoard::NUM_VERTICES] = {};
                auto newpos = str_vtx;
                auto n_vtx = 0;
                do {
                    // Check whether can capture the stone at the breathing point of opponent's stones.
                    for (auto d = 0; d < 4; d++) {
                        n_vtx = state->board.get_state_neighbor(newpos, d);
                        if (state->board.get_state(n_vtx) != opp ||
                            state->board.get_liberties(n_vtx) != 1) {
                            continue;
                        }
                        if (capture_checked[state->board.get_parent_stone(n_vtx)]) {
                            continue;
                        }
                        capture_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
                        auto capture_liberty_pos = state->board.get_liberty_pos(1, n_vtx);
                        if (!breath_checked[capture_liberty_pos[0]]) {
                            breath_checked[capture_liberty_pos[0]] = CHECKED;
                            if (state->is_move_legal(pla, capture_liberty_pos[0])) {
                                if (static_cast<size_t>(start + moveListLen) >= buf.size()) {
                                    stackIdx--;
                                    while (stackIdx >= 0) {
                                        state->undo_move();
                                        stackIdx--;
                                    }
                                    return DEAD;
                                }
                                buf[start + moveListLen] = capture_liberty_pos[0];
                                moveListLen++;
                            }
                        }
                    }
                    newpos = state->board.get_next_stone(newpos);
                } while (newpos != str_vtx);

                // Check for escape routes.
                auto escape_liberty_pos = state->board.get_liberty_pos(1, str_vtx);
                if (!breath_checked[escape_liberty_pos[0]]) {
                    if (static_cast<size_t>(start + moveListLen) >= buf.size()) {
                        stackIdx--;
                        while (stackIdx >= 0) {
                            state->undo_move();
                            stackIdx--;
                        }
                        return DEAD;
                    }
                    buf[start + moveListLen] = escape_liberty_pos[0];
                    moveListLen++;
                }

                // Check if the target stones have escaped at this point.
                int lowerBoundLibs;
                int upperBoundLibs;
                state->board.get_bound_num_liberties_after_play(
                    buf[start + moveListLen - 1],
                    pla,
                    lowerBoundLibs,
                    upperBoundLibs);
                if (lowerBoundLibs >= 3) {
                    // Defender immediately wins if there are provably enough libs
                    returnValue = ALIVE;
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }
                if (moveListLen == 1 && upperBoundLibs <= 1) {
                    // Attacker immediately wins if defender has not enough libs and there are no alternatives
                    returnValue = DEAD;
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }
            } else {
                // Chaser check
                moveListLen = 2;
                auto chase_liberty_pos = state->board.get_liberty_pos(2, str_vtx);
                if (static_cast<size_t>(start + moveListLen) >= buf.size()) {
                    stackIdx--;
                    while (stackIdx >= 0) {
                        state->undo_move();
                        stackIdx--;
                    }
                    return DEAD;
                }
                buf[start] = chase_liberty_pos[0];
                buf[start + 1] = chase_liberty_pos[1];
                int num_libs0 = 0;
                int num_libs1 = 0;
                auto n_vtx = 0;
                for (auto d = 0; d < 4; d++) {
                    n_vtx = state->board.get_state_neighbor(chase_liberty_pos[0], d);
                    if (state->board.get_state(n_vtx) == FastBoard::EMPTY) {
                        num_libs0++;
                    }
                    n_vtx = state->board.get_state_neighbor(chase_liberty_pos[1], d);
                    if (state->board.get_state(n_vtx) == FastBoard::EMPTY) {
                        num_libs1++;
                    }
                }
                if (num_libs0 == 0 && num_libs1 == 0 &&
                    state->board.would_be_ko_capture(buf[start], opp) &&
                    state->board.would_be_ko_capture(buf[start+1], opp)) {
                    // Both possible moves have no space above, below, left or right
                    // and can only take one stone.
                    auto after_libs = 0;
                    if (state->is_move_legal(opp, chase_liberty_pos[0])) {
                        state->play_move(opp, chase_liberty_pos[0]);
                        after_libs = state->board.get_liberties(chase_liberty_pos[0]);
                        state->undo_move();
                    }
                    if (after_libs <= 2 &&
                        state->is_move_legal(opp, chase_liberty_pos[1])) {
                        state->play_move(opp, chase_liberty_pos[1]);
                        after_libs = state->board.get_liberties(chase_liberty_pos[1]);
                        state->undo_move();
                        if (after_libs <= 2) {
                            // Both candidates have 2 or less breathing points after playing.
                            if (!state->board.has_liberty_gaining_captures(str_vtx)) {
                                returnValue = DEAD;
                                returnedFromDeeper = true;
                                stackIdx--;
                                continue;
                            }
                        }
                    }
                }
                if (buf[start] == buf[start + 1] - (BOARD_SIZE + 1) ||
                    buf[start] == buf[start + 1] - 1 ||
                    buf[start] == buf[start + 1] + 1 ||
                    buf[start] == buf[start + 1] + (BOARD_SIZE + 1)) {
                    // The two breathing points are not adjacent vertically or horizontally.
                    if (num_libs0 >= 3 && num_libs1 >= 3) {
                        // No matter which way move it, it won't be captured.
                        returnValue = ALIVE;
                        returnedFromDeeper = true;
                        stackIdx--;
                        continue;
                    } else if (num_libs0 >= 3) {
                        // If move it into libs0, it may be captured.
                        moveListLen = 1;
                    } else if (num_libs1 >= 3) {
                        // If move it into libs1, it may be captured.
                        buf[start] = buf[start + 1];
                        moveListLen = 1;
                    }
                }
                if (moveListLen > 1) {
                    int num_libsX2 = 0;
                    for (auto d = 0; d < 4; d++) {
                        auto n_vtx = state->board.get_state_neighbor(buf[start], d);
                        if (state->board.get_state(n_vtx) == pla) {
                            int libs = state->board.get_liberties(n_vtx);
                            if (libs > 1) {
                                num_libsX2 += libs * 2 - 3;
                            }
                        }
                    }
                    num_libs0 = num_libs0 * 2 + num_libsX2;
                    num_libsX2 = 0;
                    for (auto d = 0; d < 4; d++) {
                        auto n_vtx = state->board.get_state_neighbor(buf[start + 1], d);
                        if (state->board.get_state(n_vtx) == pla) {
                            int libs = state->board.get_liberties(n_vtx);
                            if (libs > 1) {
                                num_libsX2 += libs * 2 - 3;
                            }
                        }
                    }
                    num_libs1 = num_libs1 * 2 + num_libsX2;
                    if (num_libs1 > num_libs0) {
                        int tmp = buf[start];
                        buf[start] = buf[start + 1];
                        buf[start + 1] = tmp;
                    }
                }
            }
            moveListLens[stackIdx] = moveListLen;
            moveListCur[stackIdx] = 0;

        } else {
            // we returned from a deeper level (or the same level, via illegal move)
            if (returnedFromDeeper) {
                // If we returned from deeper we need to undo the move we made
                state->undo_move();
            }
            if (isDefender && returnValue == ALIVE) {
                returnedFromDeeper = true;
                stackIdx--;
                continue;
            }
            if (!isDefender && returnValue == DEAD) {
                returnedFromDeeper = true;
                stackIdx--;
                continue;
            }
            moveListCur[stackIdx]++;
        }
        // If there is no next move to search, then we lose.
        if (moveListCur[stackIdx] >= moveListLens[stackIdx]) {
            // For a defender, that means a ladder capture.
            // For an attacker, that means no ladder capture found.
            returnValue = isDefender;
            returnedFromDeeper = true;
            stackIdx--;
            continue;
        }
        int move = buf[moveListStarts[stackIdx] + moveListCur[stackIdx]];
        FastBoard::vertex_t p = (isDefender ? pla : opp);
        // Illegal move - treat it the same as a failed move, but don't return up a level so that we
        // loop again and just try the next move.
        if (!state->is_move_legal(p, move)) {
            returnValue = isDefender;
            returnedFromDeeper = false;
            continue;
        }
        state->play_move(p, move);
        searchNodeCount++;
        stackIdx++;
        moveListCur[stackIdx] = -1;
        moveListStarts[stackIdx] = moveListStarts[stackIdx-1] + moveListLens[stackIdx-1];
        moveListLens[stackIdx] = 0;
        depth = std::max(depth, stackIdx);
    }
}

void LadderDetection(
    GameState* const state,
    char* const ladder_pos,
    const std::array<float, NUM_INTERSECTIONS> &policy,
    const float &ladder_min_policy,
    const int &check_nodes
    )
{
    const auto turn_color = state->board.get_to_move();
    const auto opponent_color = FLIP_COLOR(turn_color);

    bool root_ko = (state->m_komove != FastBoard::NO_VERTEX);
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
                && state->board.get_liberties(vertex) == 2
                && !root_ko) {
                auto liberty_pos = state->board.get_liberty_pos(2, vertex);
                bool move0Works = false;
                bool move1Works = false;
                int depth0 = 0;
                int depth1 = 0;
                if (state->is_move_legal(opponent_color, liberty_pos[0])) {
                    auto state_copy = std::make_unique<GameState>(state);
                    state_copy->play_move(opponent_color, liberty_pos[0]);
#ifndef NDEBUG
                    const Time start;
#endif
                    move0Works = LadderExploration(state_copy, vertex, depth0);
#ifndef NDEBUG
                    const Time end;
                    const auto elapsed = Time::timediff_seconds(start, end);
                    if (elapsed > 0.1) {
                        auto turn_vertex = state->move_to_text(vertex);
                        myprintf_error(
                            "LadderDetection escape check1 time over:%f"
                            " seconds turn pos:%s depth:%d\n",
                            elapsed, turn_vertex.c_str(), depth0);
                        std::exit(1);
                    }
#endif
                    if (move0Works == DEAD) {
                        ladder_pos[i] = depth0;
                    }
                }
                if (state->is_move_legal(opponent_color, liberty_pos[1])) {
                    auto state_copy = std::make_unique<GameState>(state);
                    state_copy->play_move(opponent_color, liberty_pos[1]);
#ifndef NDEBUG
                    const Time start;
#endif
                    move1Works = LadderExploration(state_copy, vertex, depth1);
#ifndef NDEBUG
                    const Time end;
                    const auto elapsed = Time::timediff_seconds(start, end);
                    if (elapsed > 0.1) {
                        auto turn_vertex = state->move_to_text(vertex);
                        myprintf_error(
                            "LadderDetection escape check2 time over:%f"
                            " seconds turn pos:%s depth:%d\n",
                            elapsed, turn_vertex.c_str(), depth1);
                        std::exit(1);
                    }
#endif
                    if (move1Works == DEAD) {
                        ladder_pos[i] = std::min(static_cast<char>(depth1), ladder_pos[i]);
                    }
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
            int depth = 0;
            auto state_copy = std::make_unique<GameState>(state);
#ifndef NDEBUG
            const Time start;
#endif
            bool laddered = LadderExploration(state_copy, str_vtx, depth);
#ifndef NDEBUG
            const Time end;
            const auto elapsed = Time::timediff_seconds(start, end);
            if (elapsed > 0.1) {
                auto turn_vertex = state->move_to_text(vertex);
                myprintf_error(
                    "LadderDetection chase check time over:%f"
                    " seconds turn pos:%s depth:%d\n",
                    elapsed, turn_vertex.c_str(), depth);
                std::exit(1);
            }
#endif
            if (laddered == ALIVE) {
                ladder_pos[i] = -depth;
            }
        }
        state->undo_move();
    }
}

bool IsLadderRoot(
    GameState* const state,
    const int &move_vertex
    )
{
    if (cfg_ladder_chase == chase_t::EVERY || cfg_ladder_offense < 1) {
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
        int depth = 0;
        auto state_copy = std::make_unique<GameState>(state);
        bool laddered = LadderExploration(state_copy, str_vtx, depth);
        if (laddered == ALIVE) {
            if (depth >= cfg_ladder_offense) {
                state->undo_move();
                return true;
            }
        }
    }
    state->undo_move();
    return false;
}
