#ifndef _LADDER_DETECTION_H_
#define _LADDER_DETECTION_H_

#include <stack>

#include "GameState.h"

int IsLadderEscape(
    const GameState* const state,
    const int &str_vtx,
    const bool &chase = false,
    std::stack<int> *move_stack = nullptr
);

int IsLadderChase(
    const GameState* const state,
    const int &chase_vtx
);

void LadderDetection(
    const GameState* state,
    int* const ladder_pos,
    const std::array<float, NUM_INTERSECTIONS> &policy,
    const float &ladder_min_policy,
    const int &check_nodes = INT_MAX
);
#endif
