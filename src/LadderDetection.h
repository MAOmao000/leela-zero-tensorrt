#ifndef _LADDER_DETECTION_H_
#define _LADDER_DETECTION_H_

#include "GameState.h"

void LadderDetection(
    GameState* const state,
    int *ladder_pos,
    const std::array<float, NUM_INTERSECTIONS>& policy,
    const float ladder_min_policy,
    const int check_nodes = INT_MAX
);

bool IsLadderRoot(
    GameState* const state,
    const int move_vertex
);
#endif
