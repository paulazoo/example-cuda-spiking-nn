#ifndef KERNEL_STIMULUS_APPLY_H
#define KERNEL_STIMULUS_APPLY_H

#include <cstdint>

#include "stimulus_view.h"

void kernel_stimulus_apply_launch(StimulusView gpu_view,
                                  const uint32_t next_event_idx,
                                  const uint32_t num_events_to_apply);

#endif  // KERNEL_STIMULUS_APPLY_H