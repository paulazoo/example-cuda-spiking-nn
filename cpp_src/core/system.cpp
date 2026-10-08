#include "system.h"

System::System() {
}

void System::add_neuron_group(std::unique_ptr<NeuronGroup> neuron_group) {
    if (neuron_group) {
        neuron_groups_.push_back(std::move(neuron_group));
    }
}

void System::add_connection(std::unique_ptr<Connection> connection) {
    if (connection) {
        connections_.push_back(std::move(connection));
    }
}

void System::add_stimulus(std::unique_ptr<Stimulus> stimulus) {
    if (stimulus) {
        stimuli_.push_back(std::move(stimulus));
    }
}

void System::add_plasticity(std::unique_ptr<Plasticity> plasticity) {
    if (plasticity) {
        plasticities_.push_back(std::move(plasticity));
    }
}

void System::add_monitor(std::unique_ptr<Monitor> monitor) {
    if (monitor) {
        monitors_.push_back(std::move(monitor));
    }
}

void System::add_step_callback(std::function<void(uint32_t)> callback) {
    if (callback) {
        step_callbacks_.push_back(std::move(callback));
    }
}


void System::step_the_clock(uint32_t clock_step) {
    for (const std::unique_ptr<NeuronGroup>& neuron_group : neuron_groups_) {
        neuron_group->evolve();
    }

    for (const std::unique_ptr<Connection>& connection : connections_) {
        connection->propagate();
    }

    for (const std::unique_ptr<Stimulus>& stimulus : stimuli_) {
        stimulus->apply();
    }

    for (const std::unique_ptr<Plasticity>& plasticity : plasticities_) {
        plasticity->update();
    }

    for (const std::unique_ptr<Monitor>& monitor : monitors_) {
        monitor->historicize(clock_step);
    }
}

void System::run(uint32_t total_clock_steps) {
    using clock = std::chrono::steady_clock;
    const auto start_time = clock::now();
    for (uint32_t clock_step = 0; clock_step < total_clock_steps; ++clock_step) {
        for (const std::function<void(uint32_t)>& callback : step_callbacks_) {
            callback(clock_step);
        }

        step_the_clock(clock_step);

        if (clock_step % buryn::recording_interval_steps == 0) {
            for (const std::unique_ptr<Monitor>& monitor : monitors_) {
                monitor->record(clock_step);
            }
            const auto now = clock::now();
            const std::chrono::duration<float> elapsed = now - start_time;
            std::cout << "Completed clock step: " << clock_step << " | elapsed: " << elapsed.count() << "s\n";
        }
    }
}
