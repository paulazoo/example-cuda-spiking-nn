# Directories
SRC_DIR := cpp_src
BUILD_DIR := build
BIN_DIR := bin
INC_DIRS := $(wildcard $(SRC_DIR)/*/)

# Create build & bin directories if they don't exist
$(shell mkdir -p $(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%,$(INC_DIRS)) $(BIN_DIR))

# Compiler & Flags
CXX := g++
CXXVERSION := -std=c++17
INCLUDES := -I$(SRC_DIR) $(patsubst %,-I%,$(INC_DIRS))

CXXFLAGS := -O3 -march=native -fopenmp $(INCLUDES)

NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17 -arch=sm_89 $(INCLUDES) -Xcompiler "-fopenmp -march=native"

# Fail early if nvcc is unavailable
ifneq ($(shell command -v $(NVCC) >/dev/null 2>&1 && echo found),found)
$(error nvcc not found on PATH. CUDA toolchain is required for this build.)
endif

# Object Files
BURYN_OBJS := $(BUILD_DIR)/main_buryn.o \
                $(BUILD_DIR)/core/buryn_definitions.o $(BUILD_DIR)/core/system.o \
                $(BUILD_DIR)/core/config.o \
                $(BUILD_DIR)/gpu_core/gpu_simulation_state.o \
                $(BUILD_DIR)/gpu_core/kernel_init_neuron_rng_states.o \
                $(BUILD_DIR)/neuron_groups/neuron_group.o \
                $(BUILD_DIR)/neuron_groups/kernel_neuron_group_evolve.o \
                $(BUILD_DIR)/connections/connection.o \
                $(BUILD_DIR)/connections/kernel_connection_propagate.o \
                $(BUILD_DIR)/stimuli/stimulus.o \
                $(BUILD_DIR)/stimuli/kernel_stimulus_apply.o \
                $(BUILD_DIR)/plasticities/plasticity.o \
                $(BUILD_DIR)/plasticities/stdp_plasticity.o \
                $(BUILD_DIR)/plasticities/istdp_plasticity.o \
                $(BUILD_DIR)/plasticities/triplet_plasticity.o \
                $(BUILD_DIR)/plasticities/kernel_stdp_plasticity_update.o \
                $(BUILD_DIR)/plasticities/kernel_istdp_plasticity_update.o \
                $(BUILD_DIR)/plasticities/kernel_triplet_plasticity_update.o \
                $(BUILD_DIR)/monitors/monitor.o \
                $(BUILD_DIR)/monitors/spike_recorder.o \
                $(BUILD_DIR)/monitors/voltage_recorder.o \
                $(BUILD_DIR)/monitors/weight_recorder.o \
                $(BUILD_DIR)/monitors/kernel_historicize_spikes.o \
                $(BUILD_DIR)/monitors/kernel_historicize_membrane_potentials.o \
                $(BUILD_DIR)/generation/generate_stimulus_schedules.o \
                $(BUILD_DIR)/generation/generate_weights.o \
                $(BUILD_DIR)/generation/generate_location.o \
                $(BUILD_DIR)/utils/math_functions.o


# Targets
# all is for random custom changes
all:
	$(MAKE) buryn_one

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXVERSION) -c $< -o $@ $(CXXFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

buryn_one: $(BURYN_OBJS)
	$(NVCC) $(NVCCFLAGS) $(BURYN_OBJS) -o $(BIN_DIR)/main_buryn -lcublas

clean:
	$(RM) -r $(BUILD_DIR) $(BIN_DIR) ../data/buryn_data/buryn_outputs_*
	$(RM) -f *.txt

