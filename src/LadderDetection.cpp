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

static int IsLadderEscape(
    const GameState* const base_state,
    const int &str_vtx
    )
{
    auto state = std::make_unique<GameState>(base_state);
    int moveListStarts[stackSize];
    int moveListLens[stackSize];
    int moveListCur[stackSize];
    int moveListAlive[stackSize];
    std::array<int, stackSize * 2> buf;

    int max_depth = 0;
    int dead_depth = stackSize;
    bool returnValue = ALIVE;

    int stackIdx = 0;
    int searchNodeCount = 0;
    moveListCur[0] = -1;
    moveListStarts[0] = 0;
    moveListLens[0] = 0;
    moveListAlive[0] = 0;
    bool returnedFromDeeper = false;

    // Make it so that pla is always the defender
    FastBoard::vertex_t pla = state->board.get_state(str_vtx);
    FastBoard::vertex_t opp = static_cast<FastBoard::vertex_t>(FLIP_COLOR(pla));
    const auto turn_color = state->board.get_to_move();

    while (true) {
        // Returned from the root - so that's the answer
        if (stackIdx < 0) {
            if (returnValue == ALIVE) {
                return std::max(max_depth, 1);
            } else {
                return -dead_depth;
            }
        }
        // If we hit the stack limit, just consider it a failed ladder.
        // If we hit a total node count limit, then just assume it doesn't work.
        if (stackIdx >= stackSize - 1
            || searchNodeCount >= MAX_LADDER_SEARCH_NODE_BUDGET) {
            return 0;
        }

        bool isDefender;
        if (turn_color == pla) {
           isDefender = ((stackIdx % 2) == 0);
        } else {
           isDefender = ((stackIdx % 2) == 1);
        }

        // We just entered this level?
        if (moveListCur[stackIdx] == -1) {
            if (state->board.get_state(str_vtx) == FastBoard::EMPTY) {
                returnValue = DEAD;
                dead_depth = std::min(dead_depth, stackIdx);
                returnedFromDeeper = true;
                stackIdx--;
                continue;
            }
            int libs = state->board.get_liberties(str_vtx);
            // Otherwise we need to keep searching.
            // Generate the move list. Attacker and defender generate moves on the group's liberties, but only the defender
            // generates moves on surrounding capturable opposing groups.
            int start = moveListStarts[stackIdx];
            int moveListLen = 0;

            // Defender check
            if (isDefender) {
                // If we are the defender and the group has 2 liberties, we already win.
                if (libs >= 2) {
                    returnValue = ALIVE;
                    returnedFromDeeper = true;
                    stackIdx--;
                    moveListAlive[stackIdx]++;
                    continue;
                }
                // Check if can capture the stone of the surrounding opponent.
                int capture_checked[FastBoard::NUM_VERTICES] = {};
                int breath_checked[FastBoard::NUM_VERTICES] = {};
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
                                    return 0;
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
                if (!breath_checked[escape_liberty_pos[0]]
                    && state->is_move_legal(pla, escape_liberty_pos[0])) {
                    breath_checked[escape_liberty_pos[0]] = 1;
                    if (static_cast<size_t>(start + moveListLen) >= buf.size()) {
                        return 0;
                    }
                    buf[start + moveListLen] = escape_liberty_pos[0];
                    moveListLen++;
                }

                // Check if the target stones have escaped at this point.
                if (breath_checked[escape_liberty_pos[0]]) {
                    int lowerBoundLibs;
                    int upperBoundLibs;
                    state->board.get_bound_num_liberties_after_play(
                        escape_liberty_pos[0],
                        pla,
                        lowerBoundLibs,
                        upperBoundLibs);
                    // Defender immediately wins if there are provably enough libs
                    if (lowerBoundLibs >= 3) {
                        moveListAlive[stackIdx]++;
                        returnedFromDeeper = true;
                        stackIdx--;
                        returnValue = ALIVE;
                        continue;
                    }
                    // Attacker immediately wins if defender has not enough libs and there are no alternatives
                    if (moveListLen == 1 && upperBoundLibs <= 1) {
                        returnValue = DEAD;
                        dead_depth = std::min(dead_depth, stackIdx);
                        returnedFromDeeper = true;
                        stackIdx--;
                        continue;
                    }
                }

                // Is there any way to escape?
                if (moveListLen < 1) {
                    returnValue = DEAD;
                    dead_depth = std::min(dead_depth, stackIdx);
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }

            // Attacker check
            } else {
                // If we are the attacker and the group has only 1 liberty, we already win.
                if (libs <= 1) {
                    returnValue = DEAD;
                    dead_depth = std::min(dead_depth, stackIdx);
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                } else if (libs >= 3) {
                    returnedFromDeeper = true;
                    stackIdx--;
                    returnValue = ALIVE;
                    continue;
                }
                moveListLen = 2;
                auto chase_liberty_pos = state->board.get_liberty_pos(2, str_vtx);
                if (static_cast<size_t>(start + moveListLen) >= buf.size()) {
                    return 0;
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
                // If we are the attacker and we're in a double-ko death situation, then assume we win.
                // Both defender liberties must be ko mouths, connecting either ko mouth must not increase the defender's
                // liberties, and none of the attacker's surrounding stones can currently be in atari.
                // This is not complete - there are situations where the defender's connections increase liberties, or where
                // the attacker has stones in atari, but where the defender is still in inescapable atari even if they have
                // a large finite number of ko threats. But it's better than nothing.
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
                            // Both moves have 2 or less breathing points after playing.
                            if (!state->board.has_liberty_gaining_captures(str_vtx)) {
                                returnValue = DEAD;
                                dead_depth = std::min(dead_depth, stackIdx);
                                returnedFromDeeper = true;
                                stackIdx--;
                                continue;
                            }
                        }
                    }
                }
                if (buf[start] != buf[start + 1] - (BOARD_SIZE + 1) &&
                    buf[start] != buf[start + 1] - 1 &&
                    buf[start] != buf[start + 1] + 1 &&
                    buf[start] != buf[start + 1] + (BOARD_SIZE + 1)) {
                    // The two breathing points are not adjacent vertically or horizontally.
                    if (num_libs0 >= 3 && num_libs1 >= 3) {
                        // No matter which way move it, it won't be captured.
                        returnValue = ALIVE;
                        returnedFromDeeper = true;
                        stackIdx--;
                        moveListAlive[stackIdx]++;
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
                    // Swap the move order
                    num_libs1 = num_libs1 * 2 + num_libsX2;
                    if (num_libs1 > num_libs0) {
                        int tmp = buf[start];
                        buf[start] = buf[start + 1];
                        buf[start + 1] = tmp;
                    }
                }
            }
            moveListLens[stackIdx] = moveListLen;
            moveListAlive[stackIdx] = 0;
            // And indicate to begin search on the first move generated.
            moveListCur[stackIdx] = 0;

        // Else, we returned from a deeper level (or the same level, via illegal move)
        } else {
            // If we returned from deeper we need to undo the move we made
            if (returnedFromDeeper) {
                state->undo_move();
            }
            // Counting the aliving moves
            if (returnValue == ALIVE) {
                moveListAlive[stackIdx]++;
            }
            // Move on to the next move to search
            moveListCur[stackIdx]++;
        }
        // Alive or dead is determined by the results of trying all moves
        if (moveListCur[stackIdx] >= moveListLens[stackIdx]) {
            if (moveListAlive[stackIdx] < 1) {
                // No move to alive
                returnValue = DEAD;
            } else {
                if (isDefender) {
                    // There is more than one move that can alive
                    returnValue = ALIVE;
                } else {
                    if (moveListAlive[stackIdx] == moveListLens[stackIdx]) {
                        // There is no move that can capture an opponent's stone
                        returnValue = ALIVE;
                    } else {
                        // There is more than one move that can capture an opponent's stone
                        returnValue = DEAD;
                    }
                }
            }
            returnedFromDeeper = true;
            stackIdx--;
            continue;
        }
        // Otherwise we do have an next move to search. Grab it.
        int move = buf[moveListStarts[stackIdx] + moveListCur[stackIdx]];
        FastBoard::vertex_t p = (isDefender ? pla : opp);
        // Illegal move - treat it the same as a failed move, but don't return up a level so that we
        // loop again and just try the next move.
        if (!state->is_move_legal(p, move)) {
            if (isDefender) {
                returnValue = DEAD;
            } else {
                returnValue = ALIVE;
            }
            returnedFromDeeper = false;
            continue;
        }
        // Play the move!
        state->play_move(p, move);
        searchNodeCount++;
        // And recurse to the next level
        stackIdx++;
        moveListCur[stackIdx] = -1;
        moveListStarts[stackIdx] = moveListStarts[stackIdx-1] + moveListLens[stackIdx-1];
        moveListLens[stackIdx] = 0;
        moveListAlive[stackIdx] = 0;
        max_depth = std::max(max_depth, stackIdx);
    }
}

int IsLadderChase(
    const GameState* base_state,
    const int &move_vertex
    )
{
    auto state = std::make_unique<GameState>(base_state);
    if (cfg_ladder_offense < 1) {
        return 0;
    }
    auto depth = 1;
    const auto turn_color = state->board.get_to_move(); // chase
    const auto opponent_color = FLIP_COLOR(turn_color); // escape

    state->play_move(turn_color, move_vertex); // chase

    auto str_vtx = -1;
    auto liberty_vtx = -1;
    int ladder_checked[FastBoard::NUM_VERTICES] = {};
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
                    return 0;
                }
                str_vtx = n_vtx;
                liberty_vtx = liberty_pos[0];
            }
        }
    }
    if (liberty_vtx == -1) {
        return 0;
    }
    while (true) {
        state->play_move(opponent_color, liberty_vtx); // escape
        depth++;
        if (state->board.get_liberties(liberty_vtx) == 1) {
            auto one_liberty_pos = state->board.get_liberty_pos(1, liberty_vtx);
            if (state->is_move_legal(turn_color, one_liberty_pos[0])) {
                // Check if can capture the stone of the surrounding opponent.
                state->undo_move();
                int capture_checked[FastBoard::NUM_VERTICES] = {};
                int breath_checked[FastBoard::NUM_VERTICES] = {};
                auto newpos = str_vtx;
                auto n_vtx = 0;
                do {
                    // Check whether can capture the stone at the breathing point of opponent's stones.
                    for (auto d = 0; d < 4; d++) {
                        n_vtx = state->board.get_state_neighbor(newpos, d);
                        if (state->board.get_state(n_vtx) != turn_color ||
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
                            if (state->is_move_legal(opponent_color, capture_liberty_pos[0])) {
                                return depth;
                            }
                        }
                    }
                    newpos = state->board.get_next_stone(newpos);
                } while (newpos != str_vtx);
                return 0;
            } else {
                return depth;
            }
        } else if (state->board.get_liberties(liberty_vtx) >= 3) {
            return depth;
        }
        auto chase_move = -1;
        auto two_liberty_pos = state->board.get_liberty_pos(2, liberty_vtx);
        if (std::abs(two_liberty_pos[0] - liberty_vtx) == 1) {
            if ((state->board.get_state(liberty_vtx - 1) == opponent_color
                && state->board.get_state(liberty_vtx - 2) != opponent_color)
                || (state->board.get_state(liberty_vtx + 1) == opponent_color
                && state->board.get_state(liberty_vtx + 2) != opponent_color)) {
                if (state->is_move_legal(turn_color, two_liberty_pos[0])) {
                    chase_move = two_liberty_pos[0];
                }
            }
        } else if (std::abs(two_liberty_pos[0] - liberty_vtx) == BOARD_SIZE + 1) {
            if ((state->board.get_state(liberty_vtx - BOARD_SIZE - 1) == opponent_color
                && state->board.get_state(liberty_vtx - 2 * BOARD_SIZE - 2) != opponent_color)
                || (state->board.get_state(liberty_vtx + BOARD_SIZE + 1) == opponent_color
                && state->board.get_state(liberty_vtx + 2 * BOARD_SIZE + 2) != opponent_color)) {
                if (state->is_move_legal(turn_color, two_liberty_pos[0])) {
                    chase_move = two_liberty_pos[0];
                }
            }
        }
        if (std::abs(two_liberty_pos[1] - liberty_vtx) == 1) {
            if ((state->board.get_state(liberty_vtx - 1) == opponent_color
                && state->board.get_state(liberty_vtx - 2) != opponent_color)
                || (state->board.get_state(liberty_vtx + 1) == opponent_color
                && state->board.get_state(liberty_vtx + 2) != opponent_color)) {
                if (state->is_move_legal(turn_color, two_liberty_pos[1])) {
                    if (chase_move != -1) {
                        return 0;
                    }
                    chase_move = two_liberty_pos[1];
                }
            }
        } else if (std::abs(two_liberty_pos[1] - liberty_vtx) == BOARD_SIZE + 1) {
            if ((state->board.get_state(liberty_vtx - BOARD_SIZE - 1) == opponent_color
                && state->board.get_state(liberty_vtx - 2 * BOARD_SIZE - 2) != opponent_color)
                || (state->board.get_state(liberty_vtx + BOARD_SIZE + 1) == opponent_color
                && state->board.get_state(liberty_vtx + 2 * BOARD_SIZE + 2) != opponent_color)) {
                if (state->is_move_legal(turn_color, two_liberty_pos[1])) {
                    if (chase_move != -1) {
                        return 0;
                    }
                    chase_move = two_liberty_pos[1];
                }
            }
        }
        if (chase_move == -1) {
            return 0;
        }
        state->play_move(turn_color, chase_move); // chase_move
        depth++;
        if (state->board.get_state(liberty_vtx) == FastBoard::EMPTY ||
            state->board.get_liberties(liberty_vtx) != 1) {
            return 0;
        }
        auto escape_liberty_pos = state->board.get_liberty_pos(1, liberty_vtx);
        if (!state->is_move_legal(opponent_color, escape_liberty_pos[0])) {
            return 0;
        }
        liberty_vtx = escape_liberty_pos[0];
    }
}

void LadderDetection(
    const GameState* base_state,
    int* const ladder_pos,
    const std::array<float, NUM_INTERSECTIONS> &policy,
    const float &ladder_min_policy,
    const int &check_nodes
    )
{
    auto state = std::make_unique<GameState>(base_state);
    const auto turn_color = state->board.get_to_move();
    const auto opponent_color = FLIP_COLOR(turn_color);

    if (state->m_komove != FastBoard::NO_VERTEX) {
        return;
    }
    int check_nodes_count = 0;
    char ladder_checked[FastBoard::NUM_VERTICES] = {};

    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        if (check_nodes_count >= check_nodes) {
            break;
        }
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state->board.get_vertex(x, y);

        if (cfg_ladder_defense > 0 &&
            state->board.get_state(vertex) == FastBoard::EMPTY &&
            cfg_defense_stones < 1 &&
            policy[i] >= ladder_min_policy) {
            auto liberty_count = 0;
            std::array<int, 4> liberty_pos;
            for (auto d = 0; d < 4; d++) {
                auto n_vtx = state->board.get_state_neighbor(vertex, d);
                if (state->board.get_state(n_vtx) == turn_color) {
                    liberty_count = 0;
                    break;
                } else if (state->board.get_state(n_vtx) == FastBoard::EMPTY &&
                    state->is_move_legal(opponent_color, n_vtx)) {
                    liberty_pos[liberty_count] = n_vtx;
                    liberty_count++;
                }
            }
            if (liberty_count == 2 && state->is_move_legal(turn_color, vertex)) {
                check_nodes_count++;
                state->play_move(turn_color, vertex);
                auto depth = IsLadderEscape(state.get(), vertex);
                if (depth < 0) {
                    ladder_pos[i] = 1 - depth;
                }
                state->undo_move();
            }
        } else if (cfg_ladder_defense > 0 &&
            state->board.get_state(vertex) == turn_color &&
            !ladder_checked[state->board.get_parent_stone(vertex)] &&
            state->board.get_string_count(vertex) >= cfg_defense_stones &&
            state->board.get_liberties(vertex) == 1) {

            ladder_checked[state->board.get_parent_stone(vertex)] = CHECKED;
            auto liberty_pos = state->board.get_liberty_pos(1, vertex);
            auto xy = state->board.get_xy(liberty_pos[0]);
            auto move = xy.first + xy.second * BOARD_SIZE;

            if (policy[move] >= ladder_min_policy) {
                check_nodes_count++;
                if (!state->is_move_legal(turn_color, liberty_pos[0])) {
                    ladder_pos[move] = 1;
                } else {
                    auto depth = IsLadderEscape(state.get(), vertex);
                    if (depth < 0) {
                        ladder_pos[move] = -depth;
                    }
                }
            }
        }
        if (cfg_ladder_offense > 0 &&
            cfg_ladder_chase == chase_t::EVERY &&
            state->board.get_state(vertex) == FastBoard::EMPTY &&
            ladder_pos[i] == 0 &&
            policy[i] >= ladder_min_policy) {

            check_nodes_count++;
            int depth = IsLadderChase(state.get(), vertex);
            if (depth > 0) {
                ladder_pos[i] = -depth;
            }
        }
    }
}
