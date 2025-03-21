#ifndef _LADDER_DETECTION_H_
#define _LADDER_DETECTION_H_

#include "GameState.h"

int IsSimpleLadderEscape(
    const GameState* state,
    const int &move_vertex,
    const int &check_defense_stones = 1
);

int IsSimpleLadderChase(
    const GameState* state,
    const int &move_vertex,
    const int &check_offense_stones = 1

);

void LadderDetection(
    const GameState* state,
    int* const ladder_pos,
    const std::array<float, NUM_INTERSECTIONS> &policy,
    const float &ladder_min_policy,
    const int &check_nodes = INT_MAX
);

void SimpleLadderDetection(
    const GameState* state,
    int* const ladder_pos,
    const std::array<float, NUM_INTERSECTIONS> &policy,
    const float &ladder_min_policy,
    const int &check_nodes = INT_MAX
);
#endif
