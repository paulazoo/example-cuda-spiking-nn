#include "math_functions.h"

namespace math_functions {

// Euclidean distance on a 2D square lattice with periodic boundary conditions.
// Lx, Ly are the lattice extents in each dimension (e.g., number of columns/rows,
// or physical length if i/j are in continuous coordinates).
float euclidean_distance_pbc(float i, float j, float ip, float jp, float Lx, float Ly) {
    float dx = std::fabs(i - ip);
    float dy = std::fabs(j - jp);

    dx = std::min(dx, Lx - dx);  // wrap-around shortest distance
    dy = std::min(dy, Ly - dy);

    return std::sqrt(dx * dx + dy * dy);
}


}