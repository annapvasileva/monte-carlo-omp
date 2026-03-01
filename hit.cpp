#include "hit.h"

#include <cmath>

bool hit_test(float x, float y, float z) {
    return x * x * x * (x - 2) + 4 * (y * y + z * z) < 0;
}

const float *get_axis_range() {
    static float ret[6];
    ret[0] = 0.0f;
    ret[1] = 2.0f;
    ret[2] = -3.0f / 8.0f * sqrt(3.0f);
    ret[3] = 3.0f / 8.0f * sqrt(3.0f);
    ret[4] = -3.0f / 8.0f * sqrt(3.0f);
    ret[5] = 3.0f / 8.0f * sqrt(3.0f);
    
    return ret;
}
