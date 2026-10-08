#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <cstddef>
#include <string>
#include <cmath>
#include <numeric>

namespace math_functions {

float euclidean_distance_pbc(float i, float j, float ip, float jp, float Lx, float Ly);

}  // namespace math_functions

#endif  // MATH_FUNCTIONS_H
