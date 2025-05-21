# Quantum Pulse Simulation Framework

A Python framework for simulating quantum systems with pulse-driven dynamics using QuTiP, supporting multi-system interactions, batch processing, and advanced visualization.

## Key Features

### Core Components
- **QutipPulseSimulator**: Manages time evolution of multi-system quantum states
  - Batch processing for memory-efficient simulations
  - Automatic time step calculation based on pulse frequencies
  - Tensor product handling for multi-system interactions
  - State persistence with disk-based batch storage

### Pulse Sequence Construction
- **PulseSequence**: Unified interface for pulse sequence generation
  - Built-in pulse shapes:
    - Square, Sin², Gaussian, Flattop Gaussian
  - Standard interactions:
    ```
    .add_drive()         # (a + a†)^n drive
    .add_two_photon_drive()
    .add_beamsplitter()
    .add_two_photon_exchange()
    .add_second_order_beamsplitter()
    .add_trisqz()        # Cubic nonlinearity
    .add_waiting()       # Idle period
    ```
  - Custom Hamiltonian support
  - Automatic tone frequency calculation

### System Configuration
- **QuantumSystem**: Flexible quantum system definition
  - Hamiltonian composition:
    ```
    .add_harmonic_oscillator()
    .add_kerr_oscillator(Kerr)
    .add_four_wave_mixer(g4)
    .add_drive({power: strength})  # Polynomial drives
    .add_custom_hamiltonian()
    ```

### Visualization Tools
- **Wigner Function Analysis**
  - Static plots (`plot_wigner()`)
  - Animated temporal evolution (`animate_wigner()`)
  - Automatic color scale normalization
- **State Population Tracking**
  - Fock state probabilities (`plot_fock_expectations()`)
- **Pulse Sequence Visualization**
  - Interactive pulse timeline plots
  - Multi-system pulse coordination display


## Dependencies
- QuTiP 5+
- NumPy
- Matplotlib
- JAX (optional for GPU acceleration)
- tqdm
