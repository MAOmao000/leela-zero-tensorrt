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

static bool LadderEscape(
    unsigned char &depth,
    GameState *state,
    const int &str_vtx,
    const bool &escape,
    int &escape_pos);

static bool LadderChase(
    unsigned char &depth,
    GameState *state,
    const int &str_vtx,
    const bool &escape);

static bool LadderEscape(
    unsigned char &depth,
    GameState *state,
    const int &str_vtx,
    const bool &escape,
    int &escape_pos)
{
    if (escape) {
        if (depth >= 100) {
            return ALIVE;
        }
    } else {
        if (depth >= 100) {
            depth = 255;
            return ALIVE;
        }
    }
    auto escape_color = state->board.get_state(str_vtx);
    if (escape_color == FastBoard::EMPTY) {
        return DEAD;
    }
    auto num_liberty = state->board.get_liberties(str_vtx);
    if (num_liberty >= 2) {
        return ALIVE;
    }
    const auto turn_color = state->board.get_to_move();
    unsigned char max_depth_alive = 0;
    unsigned char max_depth_dead = 0;
    auto base_depth = depth;

    // Check if can capture the stone of the surrounding opponent.
    unsigned char capture_checked[FastBoard::NUM_VERTICES] = {};
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

                    max_depth_alive = std::max(depth, max_depth_alive);
                } else {
                    max_depth_dead = std::max(depth, max_depth_dead);
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
                if (escape && !base_depth) {
                    if (depth > max_depth_alive) {
                        depth = max_depth_alive;
                        return ALIVE;
                    }
                    return DEAD;
                } else {
                    depth = max_depth_alive;
                    return ALIVE;
                }
            }
            depth = std::max(depth, max_depth_dead);
            return DEAD;
        }
    }
    if (max_depth_alive) {
        depth = max_depth_alive;
        return ALIVE;
    } else if (max_depth_dead) {
        depth = max_depth_dead;
    }
    return DEAD;
}

static bool LadderChase(
    unsigned char &depth,
    GameState *state,
    const int &str_vtx,
    const bool &escape)
{
    if (escape) {
        if (depth >= 100) {
            return ALIVE;
        }
    } else {
        if (depth >= 100) {
            depth = 255;
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
    unsigned char max_depth_alive = 0;
    unsigned char min_depth_dead = 255;
    auto liberty_pos = state->board.get_liberty_pos(2, str_vtx);
    for (auto i = 0; i < 2; i++) {
        if (liberty_pos[i] && state->is_move_legal(turn_color, liberty_pos[i])) {
            state->play_move(turn_color, liberty_pos[i]);
            depth = base_depth;
            int escape_pos = -1;
            if (LadderEscape(
                ++depth,
                state,
                str_vtx,
                escape,
                escape_pos
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
    if (min_depth_dead < 255) {
        depth = min_depth_dead;
        return DEAD;
    }
    depth = std::max(depth, max_depth_alive);
    return ALIVE;
}

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
    int dead_depth = stackSize + 1;
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
                if (dead_depth <= stackSize) {
                    return -dead_depth;
                } else {
                    return 0;
                }
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
                    breath_checked[escape_liberty_pos[0]] = CHECKED;
                    if (static_cast<size_t>(start + moveListLen) >= buf.size()) {
                        return 0;
                    }
                    buf[start + moveListLen] = escape_liberty_pos[0];
                    moveListLen++;
                }

                // Check if the target stones have escaped at this point.
                if (breath_checked[escape_liberty_pos[0]]) {
                    state->play_move(pla, escape_liberty_pos[0]);
                    // Defender immediately wins if there are provably enough libs
                    if (state->board.get_liberties(escape_liberty_pos[0]) >= 3) {
                        state->undo_move();
                        returnValue = ALIVE;
                        returnedFromDeeper = true;
                        stackIdx--;
                        continue;
                    }
                    // Attacker immediately wins if defender has not enough libs and there are no alternatives
                    if (moveListLen == 1 &&
                        state->board.get_liberties(escape_liberty_pos[0]) <= 1) {
                        auto capture_liberty_pos =
                            state->board.get_liberty_pos(1, escape_liberty_pos[0]);
                        if (state->is_move_legal(opp, capture_liberty_pos[0])) {
                            state->undo_move();
                            returnValue = DEAD;
                            dead_depth = std::min(dead_depth, stackIdx);
                            returnedFromDeeper = true;
                            stackIdx--;
                            continue;
                        }
                    }
                    state->undo_move();
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
                    returnValue = ALIVE;
                    returnedFromDeeper = true;
                    stackIdx--;
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
            if (cfg_ladder_defense > 1) {
                // Counting the aliving moves
                if (returnValue == ALIVE) {
                    moveListAlive[stackIdx]++;
                }
            } else {
                // Defender has a move that is not ladder captured?
                if (isDefender && returnValue == ALIVE) {
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                // Attacker has a move that does ladder capture?
                } else if (!isDefender && returnValue == DEAD) {
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }
            }
            // Move on to the next move to search
            moveListCur[stackIdx]++;
        }
        // Alive or dead is determined by the results of trying all moves
        if (moveListCur[stackIdx] >= moveListLens[stackIdx]) {
            if (moveListAlive[stackIdx] < 1) {
                // No move to alive
                if (returnValue == ALIVE) {
                    returnValue = DEAD;
                    dead_depth = std::min(dead_depth, stackIdx);
                }
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
                        if (returnValue == ALIVE) {
                            returnValue = DEAD;
                            dead_depth = std::min(dead_depth, stackIdx);
                        }
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
                if (returnValue == ALIVE) {
                    returnValue = DEAD;
                    dead_depth = std::min(dead_depth, stackIdx);
                }
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

int IsSimpleLadderEscape(
    const GameState* base_state,
    const int &move_vertex,
    const int &check_defense_stones
    )
{
    auto state = std::make_unique<GameState>(base_state);
    const auto turn_color = state->board.get_to_move(); // Escape side
    const auto opponent_color = FLIP_COLOR(turn_color); // Chase side

    // No need to check for escape?
    if (cfg_ladder_defense < 1) {
        return 0;
    }
    // Check whether the moved stone cannot escape.
    state->play_move(turn_color, move_vertex); // Check for escape situations
    // Do not check if the number of escape stones is less than the specified number
    // or there are not two breathing points.
    auto defense_stones = state->board.get_string_count(move_vertex);
    if (defense_stones < check_defense_stones
        || state->board.get_liberties(move_vertex) != 2) {
        return 0;
    }
    // Check start.
    auto depth = 0;
    auto liberty_vtx = move_vertex; // breathing point vertex of the escape side
    while (true) {
        if (depth == 0) {
            depth++;
        } else {
            depth++;
            state->play_move(turn_color, liberty_vtx); // escape
            if (state->board.get_liberties(liberty_vtx) == 1) {
                // There is no escape route, so check to can capture the oppornent's stones.
                auto one_liberty_pos = state->board.get_liberty_pos(1, liberty_vtx);
                if (state->is_move_legal(opponent_color, one_liberty_pos[0])) {
                    state->undo_move(); // Return to the state of board before the escape
                    int capture_checked[FastBoard::NUM_VERTICES] = {};
                    int breath_checked[FastBoard::NUM_VERTICES] = {};
                    auto newpos = move_vertex;
                    auto n_vtx = 0;
                    // Check whether can capture the stone at the breathing point of opponent's stones.
                    do {
                        for (auto d = 0; d < 4; d++) {
                            n_vtx = state->board.get_state_neighbor(newpos, d);
                            if (state->board.get_state(n_vtx) != opponent_color ||
                                state->board.get_liberties(n_vtx) != 1 ||
                                capture_checked[state->board.get_parent_stone(n_vtx)]) {
                                continue;
                            }
                            capture_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
                            auto capture_liberty_pos = state->board.get_liberty_pos(1, n_vtx);
                            if (!breath_checked[capture_liberty_pos[0]]) {
                                breath_checked[capture_liberty_pos[0]] = CHECKED;
                                if (state->is_move_legal(turn_color, capture_liberty_pos[0])) {
                                    // Can escape because can capture the opponent's stones.
                                    return 0;
                                }
                            }
                        }
                        newpos = state->board.get_next_stone(newpos);
                    } while (newpos != move_vertex);
                    // Cannot escape because cannot capture the opponent's stones.
                    return depth + (defense_stones - 1) * 2;
                } else {
                    // For ko etc. the opposing player cannot place a stone on the capture position
                    // Can escape.
                    return 0;
                }
            } else if (state->board.get_liberties(liberty_vtx) >= 3) {
                // If the turning player have three or more escape routes, the turn player can escape.
                return 0;
            }
        }
        // Look for a move that will create a staircase of escape's string.
        auto chase_move = -1;
        auto two_liberty_pos = state->board.get_liberty_pos(2, liberty_vtx);
        for (auto i = 0; i < 2; i++) {
            if (std::abs(two_liberty_pos[i] - liberty_vtx) == 1) {
                if ((state->board.get_state(liberty_vtx - 1) == turn_color
                    && state->board.get_state(liberty_vtx - 2) != turn_color)
                    || (state->board.get_state(liberty_vtx + 1) == turn_color
                    && state->board.get_state(liberty_vtx + 2) != turn_color)) {
                    if (state->is_move_legal(opponent_color, two_liberty_pos[i])) {
                        chase_move = two_liberty_pos[i];
                        break;
                    }
                }
            } else if (std::abs(two_liberty_pos[i] - liberty_vtx) == BOARD_SIZE + 1) {
                if ((state->board.get_state(liberty_vtx - BOARD_SIZE - 1) == turn_color
                    && state->board.get_state(liberty_vtx - 2 * BOARD_SIZE - 2) != turn_color)
                    || (state->board.get_state(liberty_vtx + BOARD_SIZE + 1) == turn_color
                    && state->board.get_state(liberty_vtx + 2 * BOARD_SIZE + 2) != turn_color)) {
                    if (state->is_move_legal(opponent_color, two_liberty_pos[i])) {
                        chase_move = two_liberty_pos[i];
                        break;
                    }
                }
            }
        }
        if (chase_move == -1) {
            // If there is no corresponding move, play the first move found.
            chase_move = two_liberty_pos[0];
        }
        // Try the first candidate move.
        state->play_move(opponent_color, chase_move); // chase_move
        depth++;
        if (state->board.get_state(liberty_vtx) == FastBoard::EMPTY) {
            // The player cannot escape because the stone of the turning player is taken.
            return depth + (defense_stones - 1) * 2;
        } else if (state->board.get_liberties(liberty_vtx) != 1) {
            // The opposing player cannot be atari and can escape.
            return 0;
        }
        auto escape_liberty_pos = state->board.get_liberty_pos(1, liberty_vtx);
        if (!state->is_move_legal(turn_color, escape_liberty_pos[0])) {
            // The turning player cannot be atari and can escape.
            return 0;
        }
        // If it is a move that can be escaped by ladder breaker or other means,
        // try another move.
        state->play_move(turn_color, escape_liberty_pos[0]); // escape
        if (state->board.get_liberties(escape_liberty_pos[0]) < 3) {
            state->undo_move();
            liberty_vtx = escape_liberty_pos[0];
            continue;
        }
        // Try a second candidate move.
        state->undo_move();
        state->undo_move();
        if (chase_move == two_liberty_pos[0]) {
            chase_move = two_liberty_pos[1];
        } else {
            chase_move = two_liberty_pos[0];
        }
        state->play_move(opponent_color, chase_move); // chase_move
        if (state->board.get_state(liberty_vtx) == FastBoard::EMPTY) {
            // The player cannot escape because the stone of the turning player is taken.
            return depth + (defense_stones - 1) * 2;
        } else if (state->board.get_liberties(liberty_vtx) != 1) {
            // The opposing player cannot be atari and can escape.
            return 0;
        }
        escape_liberty_pos = state->board.get_liberty_pos(1, liberty_vtx);
        if (!state->is_move_legal(turn_color, escape_liberty_pos[0])) {
            // The opposing player cannot be atari and can escape.
            return 0;
        }
        liberty_vtx = escape_liberty_pos[0];
    }
}

int IsSimpleLadderChase(
    const GameState* base_state,
    const int &move_vertex,
    const int &check_offense_stones
    )
{
    auto state = std::make_unique<GameState>(base_state);
    const auto turn_color = state->board.get_to_move(); // Chase side
    const auto opponent_color = FLIP_COLOR(turn_color); // Escape side

    // No need to check for chase?
    if (cfg_ladder_offense < 1) {
        return 0;
    }
    // Can capture the opponent's stone by the moving stone?
    state->play_move(turn_color, move_vertex);
    // Check to see if there are adjacent stones of opponents
    // the turning player might be able to capture on the top, bottom, left, or right.
    auto offense_stones = 1;
    auto depth = 1;
    auto str_vtx = -1;
    auto liberty_vtx = -1;
    int ladder_checked[FastBoard::NUM_VERTICES] = {};
    for (auto d = 0; d < 4; d++) {
        auto n_vtx = state->board.get_state_neighbor(move_vertex, d);
        if (state->board.get_state(n_vtx) == opponent_color
            && !ladder_checked[state->board.get_parent_stone(n_vtx)]
            && state->board.get_string_count(n_vtx) >= check_offense_stones
            && state->board.get_liberties(n_vtx) == 1) {
            ladder_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
            auto liberty_pos = state->board.get_liberty_pos(1, n_vtx);
            if (state->is_move_legal(opponent_color, liberty_pos[0])) {
                if (liberty_vtx != -1) {
                    // If the stone is adjacent to an inescapable opponent's stone,
                    // it can be captured.
                    return 0;
                }
                str_vtx = n_vtx;
                liberty_vtx = liberty_pos[0];
                offense_stones = state->board.get_string_count(n_vtx);
            }
        }
    }
    if (liberty_vtx == -1) {
        // Not checked because there are no applicable stones around.
        return 0;
    }
    while (true) {
        depth++;
        state->play_move(opponent_color, liberty_vtx); // escape
        if (state->board.get_liberties(liberty_vtx) == 1) {
            // There is no escape route, so check to can capture the oppornent's stones.
            auto one_liberty_pos = state->board.get_liberty_pos(1, liberty_vtx);
            if (state->is_move_legal(turn_color, one_liberty_pos[0])) {
                state->undo_move(); // Return to the state of board before the escape
                int capture_checked[FastBoard::NUM_VERTICES] = {};
                int breath_checked[FastBoard::NUM_VERTICES] = {};
                auto newpos = str_vtx;
                auto n_vtx = 0;
                // Check whether can capture the stone at the breathing point of opponent's stones.
                do {
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
                                // Can escape because can capture the opponent's stones.
                                return depth + (offense_stones - 1) * 2;
                            }
                        }
                    }
                    newpos = state->board.get_next_stone(newpos);
                } while (newpos != str_vtx);
                // Cannot escape because cannot capture the opponent's stones.
                return 0;
            } else {
                // For ko etc. the turning player cannot place a stone on the capture position
                // Can escape.
                return depth + (offense_stones - 1) * 2;
            }
        } else if (state->board.get_liberties(liberty_vtx) >= 3) {
            // If the opposing player have three or more escape routes, the turn player can escape.
            return depth + (offense_stones - 1) * 2;
        }
        // Look for a move that will create a staircase of escape's string.
        auto chase_move = -1;
        auto two_liberty_pos = state->board.get_liberty_pos(2, liberty_vtx);
        for (auto i = 0; i < 2; i++) {
            if (std::abs(two_liberty_pos[i] - liberty_vtx) == 1) {
                if ((state->board.get_state(liberty_vtx - 1) == opponent_color
                    && state->board.get_state(liberty_vtx - 2) != opponent_color)
                    || (state->board.get_state(liberty_vtx + 1) == opponent_color
                    && state->board.get_state(liberty_vtx + 2) != opponent_color)) {
                    if (state->is_move_legal(turn_color, two_liberty_pos[i])) {
                        chase_move = two_liberty_pos[i];
                        break;
                    }
                }
            } else if (std::abs(two_liberty_pos[i] - liberty_vtx) == BOARD_SIZE + 1) {
                if ((state->board.get_state(liberty_vtx - BOARD_SIZE - 1) == opponent_color
                    && state->board.get_state(liberty_vtx - 2 * BOARD_SIZE - 2) != opponent_color)
                    || (state->board.get_state(liberty_vtx + BOARD_SIZE + 1) == opponent_color
                    && state->board.get_state(liberty_vtx + 2 * BOARD_SIZE + 2) != opponent_color)) {
                    if (state->is_move_legal(turn_color, two_liberty_pos[i])) {
                        chase_move = two_liberty_pos[i];
                        break;
                    }
                }
            }
        }
        if (chase_move == -1) {
            // If there is no corresponding move, play the first move found.
            chase_move = two_liberty_pos[0];
        }
        // Try the first candidate move.
        state->play_move(turn_color, chase_move); // chase_move
        depth++;
        if (state->board.get_state(liberty_vtx) == FastBoard::EMPTY) {
            // The player cannot escape because the stone of the opposing player is taken.
            return 0;
        } else if (state->board.get_liberties(liberty_vtx) != 1) {
            // The turning player cannot be atari and can escape.
            return depth + (offense_stones - 1) * 2;
        }
        auto escape_liberty_pos = state->board.get_liberty_pos(1, liberty_vtx);
        if (!state->is_move_legal(opponent_color, escape_liberty_pos[0])) {
            // The opposing player cannot be atari and can escape.
            return depth + (offense_stones - 1) * 2;
        }
        // If it is a move that can be escaped by ladder breaker or other means,
        // try another move.
        state->play_move(opponent_color, escape_liberty_pos[0]); // escape
        if (state->board.get_liberties(escape_liberty_pos[0]) < 3) {
            state->undo_move();
            liberty_vtx = escape_liberty_pos[0];
            continue;
        }
        // Try a second candidate move.
        state->undo_move();
        state->undo_move();
        if (chase_move == two_liberty_pos[0]) {
            chase_move = two_liberty_pos[1];
        } else {
            chase_move = two_liberty_pos[0];
        }
        state->play_move(turn_color, chase_move); // chase_move
        if (state->board.get_state(liberty_vtx) == FastBoard::EMPTY) {
            // The player cannot escape because the stone of the opposing player is taken.
            return 0;
        } else if (state->board.get_liberties(liberty_vtx) != 1) {
            // The turning player cannot be atari and can escape.
            return depth + (offense_stones - 1) * 2;
        }
        escape_liberty_pos = state->board.get_liberty_pos(1, liberty_vtx);
        if (!state->is_move_legal(opponent_color, escape_liberty_pos[0])) {
            // The turning player cannot be atari and can escape.
            return depth + (offense_stones - 1) * 2;
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

    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        if (check_nodes_count >= check_nodes) {
            break;
        }
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state->board.get_vertex(x, y);
        if (state->board.get_state(vertex) != FastBoard::EMPTY
            || policy[i] < ladder_min_policy
            || !state->is_move_legal(turn_color, vertex)) {
            continue;
        }
        check_nodes_count++;

        state->play_move(turn_color, vertex);
        if (cfg_recursive_ladder) {
            if (cfg_ladder_defense > 0 &&
                state->board.get_string_count(vertex) >= cfg_defense_stones &&
                state->board.get_liberties(vertex) == 2) {

                unsigned char depth = 0;
                if (LadderChase(
                    depth,
                    state.get(),
                    vertex,
                    true
                    ) == DEAD) {

/*
                    if (depth >= state->board.get_string_count(vertex) * 2) {
                        ladder_pos[i] = depth;
                    } else {
                        ladder_pos[i] = depth * 2;
                    }
*/
                    ladder_pos[i] = depth + state->board.get_string_count(vertex) * 2;
                }
            }
            if (cfg_ladder_offense < 1) {
                state->undo_move();
                continue;
            }
/*
            if (cfg_use_simple_chase) {
                state->undo_move();
                auto depth = IsSimpleLadderChase(state.get(), vertex, cfg_offense_stones);
                if (depth > 0) {
                    ladder_pos[i] = -depth;
                }
                continue;
            }
*/
            auto offense_stones = 1;
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
                        offense_stones = state->board.get_string_count(n_vtx);
                    }
                }
            }
            if (liberty_vtx != -1
//                && state->board.get_string_count(str_vtx) >= cfg_offense_stones) {
                && offense_stones >= cfg_offense_stones) {
                unsigned char depth = 0;
                if (LadderEscape(
                    depth,
                    state.get(),
                    str_vtx,
                    false,
                    liberty_vtx
                    ) == ALIVE) {

/*
                    if (depth >= state->board.get_string_count(str_vtx) * 2) {
                        ladder_pos[i] = -depth;
                    } else {
                        ladder_pos[i] = -depth * 2;
                    }
*/
                    ladder_pos[i] = -depth - offense_stones * 2;
                }
            }
            state->undo_move();
            continue;
        }
        if (cfg_ladder_defense > 0 &&
            state->board.get_string_count(vertex) >= cfg_defense_stones &&
            state->board.get_liberties(vertex) == 2) {

            auto depth = IsLadderEscape(state.get(), vertex);
            if (depth < 0) {
                ladder_pos[i] = 1 - (depth - state->board.get_string_count(vertex) * 2);
            }
        }
        if (cfg_ladder_offense < 1) {
            state->undo_move();
            continue;
        }
/*
        if (cfg_use_simple_chase) {
            state->undo_move();
            auto depth = IsSimpleLadderChase(state.get(), vertex, cfg_offense_stones);
            if (depth > 0) {
                ladder_pos[i] = -depth;
            }
            continue;
        }
*/
        std::array<int, 4> str_vtx = {-1, -1, -1, -1};
        std::array<int, 4> liberty_vtx = {-1, -1, -1, -1};
        auto opponent_num = 0;
        char ladder_checked[FastBoard::NUM_VERTICES] = {};
        for (auto d = 0; d < 4; d++) {
            auto n_vtx = state->board.get_state_neighbor(vertex, d);
            if (state->board.get_state(n_vtx) == opponent_color
                && !ladder_checked[state->board.get_parent_stone(n_vtx)]
                && state->board.get_liberties(n_vtx) == 1) {
                ladder_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
                auto liberty_pos = state->board.get_liberty_pos(1, n_vtx);
                if (state->is_move_legal(opponent_color, liberty_pos[0])) {
                    str_vtx[opponent_num] = n_vtx;
                    liberty_vtx[opponent_num] = liberty_pos[0];
                    opponent_num++;
                }
            }
        }
        for (auto opp_i = 0; opp_i < opponent_num; opp_i++) {
            if (state->board.get_string_count(str_vtx[opp_i]) >= cfg_offense_stones) {
                state->play_move(opponent_color, liberty_vtx[opp_i]);
                if (state->board.get_liberties(liberty_vtx[opp_i]) == 2) {
                    auto depth = IsLadderEscape(state.get(), str_vtx[opp_i]);
                    if (depth < 0 && ladder_pos[i] < 0) {
                        ladder_pos[i] = 0;
                        state->undo_move();
                        break;
                    } else if (depth > 0 && ladder_pos[i] <= 0) {
                        depth += state->board.get_string_count(str_vtx[opp_i]) * 2;
                        ladder_pos[i] = std::min(-depth, ladder_pos[i]);
                    }
                }
                state->undo_move();
            }
        }
        state->undo_move();
    }
}

void SimpleLadderDetection(
    const GameState* base_state,
    int* const ladder_pos,
    const std::array<float, NUM_INTERSECTIONS> &policy,
    const float &ladder_min_policy,
    const int &check_nodes
    )
{
    auto state = std::make_unique<GameState>(base_state);
    const auto turn_color = state->board.get_to_move();

    if (state->m_komove != FastBoard::NO_VERTEX) {
        return;
    }
    int check_nodes_count = 0;

    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        if (check_nodes_count >= check_nodes) {
            break;
        }
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state->board.get_vertex(x, y);
        if (state->board.get_state(vertex) != FastBoard::EMPTY
            || policy[i] < ladder_min_policy
            || !state->is_move_legal(turn_color, vertex)) {
            continue;
        }
        check_nodes_count++;
        auto depth = IsSimpleLadderEscape(state.get(), vertex, cfg_defense_stones);
        if (depth > 0) {
#ifndef NDEBUG
            auto check_vertex = state->move_to_text(vertex);
            myprintf_error("escape %s depth:%d\n", check_vertex.c_str(), depth);
#endif
            ladder_pos[i] = depth;
        }

        if (cfg_ladder_offense < 1) {
            depth = IsSimpleLadderChase(state.get(), vertex, cfg_offense_stones);
            if (depth > 0) {
#ifndef NDEBUG
                auto check_vertex = state->move_to_text(vertex);
                myprintf_error("chase %s depth:%d\n", check_vertex.c_str(), depth);
#endif
                ladder_pos[i] = -depth;
            }
        }
    }
}
