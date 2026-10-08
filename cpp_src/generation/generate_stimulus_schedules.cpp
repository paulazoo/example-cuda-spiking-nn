#include "generate_stimulus_schedules.h"

// NOTE: ChatGPT coded this
namespace generate_stimulus_schedules {

static bool write_u32_lines(const std::filesystem::path& p, const std::vector<uint32_t>& v) {
    std::ofstream out(p);
    if (!out.is_open()) return false;
    for (uint32_t i = 0; i < v.size(); ++i) out << v[i] << '\n';
    return true;
}

static bool write_f32_lines(const std::filesystem::path& p, const std::vector<float>& v) {
    std::ofstream out(p);
    if (!out.is_open()) return false;
    for (uint32_t i = 0; i < v.size(); ++i) out << v[i] << '\n';
    return true;
}

StimulusGenerationResult partial_group_multiple_square_waves(
    const std::string& folder_name,
    StimulusGenerationParams& params) {

    StimulusGenerationResult result;

    // Validate selected neuron parameter sizes
    if (params.selected_input_ids.size() != params.selected_input_post_group_offsets.size()) {
        std::cerr << "Selected neuron vectors must have the same size: "
                  << "selected_input_ids=" << params.selected_input_ids.size()
                  << ", selected_input_post_group_offsets=" << params.selected_input_post_group_offsets.size()
                  << "\n";
        return result;
    }

    // 1) Validate wave parameter sizes
    if (params.start_clock_steps.size() != params.end_clock_steps.size()) {
        std::cerr << "Multi square wave parameter vectors must have the same size: "
                  << "start_clock_steps=" << params.start_clock_steps.size()
                  << ", end_clock_steps=" << params.end_clock_steps.size() << "\n";
        return result;
    }

    if (params.stimulus_input_values.size() != params.start_clock_steps.size()) {
        if (params.stimulus_input_values.size() == 1) {
            // Broadcast single value to all waves
            params.stimulus_input_values.resize(params.start_clock_steps.size(), params.stimulus_input_values[0]);
        } else {
            std::cerr << "Multi square wave parameter vectors must have the same size: "
                    << "stimulus_input_values=" << params.stimulus_input_values.size()
                    << ", start_clock_steps=" << params.start_clock_steps.size() << "\n";
            return result;
        }
    }

    for (uint32_t i = 0; i < params.start_clock_steps.size(); ++i) {
        if (params.start_clock_steps[i] > params.end_clock_steps[i]) {
            std::cerr << "Square wave start has to be before end: start_clock_step ("
                      << params.start_clock_steps[i] << ") > end_clock_step ("
                      << params.end_clock_steps[i] << ")\n";
            return result;
        }
    }

    // 2) Use explicit simulation length
    // Interpretation: params.max_timestep = number of timesteps (steps are [0, max_timestep-1]).
    if (params.max_timestep == 0) {
        std::cerr << "params.max_timestep must be > 0\n";
        return result;
    }
    const uint32_t T = static_cast<uint32_t>(params.max_timestep);

    // Prepare output folder
    std::filesystem::path schedules_folder(folder_name);
    std::error_code ec;
    std::filesystem::create_directories(schedules_folder, ec);
    if (ec) {
        std::cerr << "Failed to create stimulus schedules folder: " << schedules_folder << "\n";
        return result;
    }

    // Selected neurons: each neuron is identified by (post_group_offset, neuron_id)
    std::vector<std::pair<uint32_t, uint32_t>> selected; // (post_group_offset, neuron_id)
    selected.reserve(params.selected_input_ids.size());

    std::unordered_set<uint64_t> seen;
    seen.reserve(params.selected_input_ids.size());

    for (uint32_t i = 0; i < params.selected_input_ids.size(); ++i) {
        const uint32_t neuron_id = params.selected_input_ids[i];
        const uint32_t post_group_offset = params.selected_input_post_group_offsets[i];
        const uint64_t key = (static_cast<uint64_t>(post_group_offset) << 32) | static_cast<uint64_t>(neuron_id);
        if (seen.insert(key).second) {
            selected.emplace_back(post_group_offset, neuron_id);
        }
    }

    if (selected.empty()) {
        // Still save empty vectors with correct timestep size
        result.timestep_event_counts.assign(T, 0u);
        (void)write_u32_lines(schedules_folder / "timestep_event_counts.txt", result.timestep_event_counts);
        (void)write_u32_lines(schedules_folder / "stimulus_event_neuron_ids.txt", result.neuron_ids);
        (void)write_u32_lines(schedules_folder / "stimulus_event_post_group_offsets.txt", result.post_group_offsets);
        (void)write_f32_lines(schedules_folder / "stimulus_event_delta_amplitudes.txt", result.amplitude_changes);
        return result;
    }

    // Build raw events: +amp at start, -amp at end+1 (inclusive square wave semantics)
    // Keep only events with 0 <= t < T.
    struct EventWithGroup {
        uint32_t t;
        uint32_t post_group_offset;
        uint32_t neuron;
        float delta;
    };

    std::vector<EventWithGroup> events;
    const uint32_t num_waves = static_cast<uint32_t>(params.start_clock_steps.size());
    events.reserve(2 * num_waves * static_cast<uint32_t>(selected.size()));

    for (const auto& sel : selected) {
        const uint32_t post_group_offset = sel.first;
        const uint32_t nid = sel.second;

        for (uint32_t i = 0; i < num_waves; ++i) {
            const uint32_t start = params.start_clock_steps[i];
            const uint32_t end_inclusive = params.end_clock_steps[i];
            const float amp = params.stimulus_input_values[i];

            // Event at start
            if (static_cast<uint32_t>(start) < T) {
                events.push_back(EventWithGroup{start, post_group_offset, nid, +amp});
            }

            // Event at end+1 to turn off after inclusive end
            const uint32_t off_t = end_inclusive + 1;
            if (static_cast<uint32_t>(off_t) < T) {
                events.push_back(EventWithGroup{off_t, post_group_offset, nid, -amp});
            }
        }
    }

    // Sort by (t, post_group_offset, neuron) so events for a timestep are contiguous and mergeable
    std::sort(events.begin(), events.end(), [](const EventWithGroup& a, const EventWithGroup& b) {
        if (a.t != b.t) return a.t < b.t;
        if (a.post_group_offset != b.post_group_offset) return a.post_group_offset < b.post_group_offset;
        return a.neuron < b.neuron;
    });

    // 3) Enforce: at each timestep, each (post_group_offset, neuron) has at most 1 event.
    // If duplicates (t, post_group_offset, neuron) detected, raise error
    std::vector<EventWithGroup> merged;
    merged.reserve(events.size());

    for (const auto& e : events) {
        if (!merged.empty() &&
            merged.back().t == e.t &&
            merged.back().post_group_offset == e.post_group_offset &&
            merged.back().neuron == e.neuron)
        {
            std::cerr << "Error: duplicate events detected for time step " << e.t
                      << ", post_group_offset " << e.post_group_offset
                      << ", neuron_id " << e.neuron << "\n";
            std::exit(1);
        } else {
            merged.push_back(e);
        }
    }

    // Optionally drop zero-delta events after merging
    constexpr float eps = 0.0f; // set to e.g. 1e-8f if desired
    if (eps > 0.0f) {
        uint32_t w = 0;
        for (uint32_t i = 0; i < merged.size(); ++i) {
            if (std::abs(merged[i].delta) > eps) merged[w++] = merged[i];
        }
        merged.resize(w);
    }

    // Build timestep_event_counts (length T)
    result.timestep_event_counts.assign(T, 0u);
    for (const auto& e : merged) {
        const uint32_t t = static_cast<uint32_t>(e.t);
        if (t < T) {
            result.timestep_event_counts[t] += 1u;
        }
    }

    // Build flat per-event arrays in time-sorted order
    result.neuron_ids.reserve(merged.size());
    result.post_group_offsets.reserve(merged.size());
    result.amplitude_changes.reserve(merged.size());
    for (const auto& e : merged) {
        result.neuron_ids.push_back(e.neuron);
        result.post_group_offsets.push_back(e.post_group_offset);
        result.amplitude_changes.push_back(e.delta);
    }

    // Save the requested 1D vectors
    {
        const bool ok1 = write_u32_lines(schedules_folder / "timestep_event_counts.txt",
                                         result.timestep_event_counts);
        const bool ok2 = write_u32_lines(schedules_folder / "stimulus_event_neuron_ids.txt",
                                         result.neuron_ids);
        const bool ok3 = write_u32_lines(schedules_folder / "stimulus_event_post_group_offsets.txt",
                                         result.post_group_offsets);
        const bool ok4 = write_f32_lines(schedules_folder / "stimulus_event_delta_amplitudes.txt",
                                         result.amplitude_changes);

        if (!ok1) std::cerr << "Failed to write timestep_event_counts.txt\n";
        if (!ok2) std::cerr << "Failed to write stimulus_event_neuron_ids.txt\n";
        if (!ok3) std::cerr << "Failed to write stimulus_event_post_group_offsets.txt\n";
        if (!ok4) std::cerr << "Failed to write stimulus_event_delta_amplitudes.txt\n";
    }

    // Optional: single combined file for debugging
    {
        std::ofstream out(schedules_folder / "stimulus_events_compact.txt");
        if (out.is_open()) {
            out << "# idx time_step post_group_offset neuron_id delta\n";
            uint32_t idx = 0;
            for (const auto& e : merged) {
                out << idx++ << " " << e.t << " " << e.post_group_offset << " " << e.neuron << " " << e.delta << "\n";
            }
        }
    }

    return result;
}

} // namespace generate_stimulus_schedules