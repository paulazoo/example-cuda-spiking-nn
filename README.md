# Docs
- environment requirements:
  - python 3.11.6
  - pip install numpy ipykernel matplotlib opencv-python


# System
![](./resources/system_page0.jpg)
- In System::step_the_clock, the sequence is: evolve neurons → propagate connections → propagate stimuli → update plasticity → record monitors. Note that this means stimuli set by Stimulus::propagate_stimulus() affect the next step's evolve().

# Neuron Groups
![](./resources/neuron_groups_page0.jpg)

### distance planning
90 x 90 = 8100 grid points for about a 1mm square, then each 9 side grid points is a distance of 100um.
(according to 8000 excitatory neurons in a L2/3 1mmx1mm and 100um deep volume).
Using $P_K^\lambda({i,j}) \sim C\exp(\frac{-d_{ij}^2}{S})=N(0, \sqrt{0.5S})$


# Connections
![](./resources/connections_page0.jpg)

### kernel_connection_propagate for $\sum_\text{pre} (\text{synapse})$
- weight matrix $\bold{\underline{A}}$ in GPU is row-major like weight_matrix[pre_neuron_id * num_post_neurons + post_neuron_id]
- cuBLAS expects column-major order so re-interpret weight_matrix as column-major matrix A with shape [post, pre] such that $\bold{\underline{A}} = \bold{\underline{W}}^T$ (by using option CUBLAS_OP_N)
- each new postsynaptic conductance is $\Delta_{\text{from this connection}} g_{\text{post neuron}} = \sum_\text{pre} (\text{synapse}) = \sum_\text{pre} W[\text{pre}, \text{post}] x[\text{pre}] = \sum_\text{pre} A[\text{post}, \text{pre}] x[\text{pre}] = \bold{\underline{A}}\vec{x}$ where $x[\text{pre}]$ is the computed postsynaptic conductance for that pre neuron.
- GEMV calculates $\vec{y} = \alpha \bold{\underline{A}}\vec{x} + \beta \vec{y}$ so we will use: $\vec{y} =$ post_group's g (either excitatory or inhibitory depending on pre_group) and $\vec{x} =$ summed_postsynaptic_conductances
- cublasSgemv:
    - cuBLAS handle
    - CUBLAS_OP_N cuBLAS expects column-major order so re-interpret weight_matrix as column-major matrix A with shape [post, pre]
    - m=num_post_neurons number of rows of A
    - n=num_pre_neurons number of columns of A
    - alpha=1
    - A=weight_matrix
    - lda=post leading dimension of A (number of rows in column-major order)
    - x=summed_postsynaptic_conductances of shape [pre, 1]
    - incx=1 increment for elements of x (not 1 means every incx-th element is used)
    - beta=1
    - y=target_post_group_g
    - incy=1 increment for elements of y (not 1 means every incy-th element is used)
- Note that inhibitory weight matrix values are still positive. Instead, to have inhibitory synapses, the connection must have pre_group_inhibitory bool set to true, and the inhibition occurs via adding to g_inhibitory_ (using add_g_inhibitory()) of the postsynaptic neuron group.

### postsynaptic conductance values
- $G^{\lambda}(t)=(\exp(-t/\tau_d^{\lambda}) - \exp(-t/\tau_r^{\lambda})) * (N)$
- $G^{\lambda}(t)=(\exp(-t/0.002) - \exp(-t/0.0003)) * (1.7)$
- $G^E(t) \approx 1.0 \text{ms}^{-1}$ @ $t=0.6\text{ms}$
- $K_{ij}^E=1.0*10^{-9}$ for large synapse
- $K^E*G^E \approx 1.0 \text{nS}$ for sort of close to biologically realistic magnitude (around 0.1-1nS)


# Plasticities
![](./resources/plasticities_page0.jpg)
![](./resources/plasticities_page1.jpg)

# Gpu Simulation State
![](./resources/gpu_simulation_state_page0.jpg)
![](./resources/gpu_simulation_state_page1.jpg)


# References
- https://www.youtube.com/watch?v=2NgpYFdsduY&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU
- https://www.youtube.com/watch?v=h9Z4oGN89MU threads, blocks, grid, cores, warps, SMs
- https://www.youtube.com/watch?v=MVutNZaNTkM&list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU&index=7 cuRAND and cuBLAS
- https://forums.developer.nvidia.com/t/undefined-reference-to-cublascreate-v2/23799/3
