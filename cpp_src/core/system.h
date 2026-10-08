#ifndef SYSTEM_H
#define SYSTEM_H

#include <functional>
#include <memory>
#include <vector>
#include <iostream>
#include <chrono>

#include "buryn_definitions.h"
#include "connection.h"
#include "neuron_group.h"
#include "stimulus.h"
#include "plasticity.h"
#include "monitor.h"

class System {
public:
    System();
    ~System() = default;
    void add_neuron_group(std::unique_ptr<NeuronGroup> neuron_group);
    void add_connection(std::unique_ptr<Connection> connection);
    void add_stimulus(std::unique_ptr<Stimulus> stimulus);
    void add_plasticity(std::unique_ptr<Plasticity> plasticity);
    void add_monitor(std::unique_ptr<Monitor> monitor);
    void add_step_callback(std::function<void(uint32_t)> callback);

    void step_the_clock(uint32_t clock_step);
    void run(uint32_t total_clock_steps);

    const std::vector<std::unique_ptr<NeuronGroup>>& neuron_groups() const { return neuron_groups_; }
    const std::vector<std::unique_ptr<Connection>>& connections() const { return connections_; }
    const std::vector<std::unique_ptr<Monitor>>& monitors() const { return monitors_; }
    const std::vector<std::function<void(uint32_t)>>& step_callbacks() const { return step_callbacks_; }

private:
    std::vector<std::unique_ptr<NeuronGroup>> neuron_groups_;
    std::vector<std::unique_ptr<Connection>> connections_;
    std::vector<std::unique_ptr<Stimulus>> stimuli_;
    std::vector<std::unique_ptr<Plasticity>> plasticities_;
    std::vector<std::unique_ptr<Monitor>> monitors_;
    std::vector<std::function<void(uint32_t)>> step_callbacks_;
};

#endif  // SYSTEM_H
