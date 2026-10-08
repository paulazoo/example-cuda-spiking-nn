#ifndef KERNEL_CONNECTION_PROPAGATE_H
#define KERNEL_CONNECTION_PROPAGATE_H

#include <cmath>
#include <cstdint>

#include "connection_params.h"
#include "connection_view.h"

void kernel_connection_propagate_launch(ConnectionView gpu_view);

#endif  // KERNEL_CONNECTION_PROPAGATE_H