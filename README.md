# NIXL Benchmark Suite

This repository contains benchmark scripts and examples for testing NVIDIA's NIXL (NVIDIA Interconnect eXchange Library) framework, which provides high-performance data transfer capabilities for distributed computing environments.

## Files Description

### Core Scripts

- **`with_nxl_ucx.py`** - Comprehensive NIXL benchmark using UCX backend that demonstrates:
  - Agent setup and configuration with UCX transport
  - Memory registration and management for high-performance data transfer
  - Bidirectional data transfer operations (READ/WRITE) between initiator and target agents
  - Inter-agent notification system for coordination
  - Proper resource cleanup and memory deallocation
  - Performance validation through multiple transfer cycles
  - **Communication**: CPU-to-CPU communication using DRAM memory (no GPU involved)
  - **Data Transfer**: 512 bytes per transfer (2 × 256-byte buffers) with 3 total transfers (~1.5 KB total)

- **`without_nxl.py`** - Baseline benchmark without NIXL for comparison

- **`nixl_gds_example.py`** - Example demonstrating NIXL with GPU Direct Storage integration
  - **Communication**: Disk-to-GPU communication using VRAM memory (no CPU involved)
  - **Data Transfer**: 5 GB test input file with buffer set to 5 MB
### Setup Scripts

- **`install_gds.sh`** - Installs NVIDIA GDS with proper repository setup
- **`install_gds_alt.sh`** - Alternative installation script with fallback options
- **`setup_venv.sh`** - Sets up Python virtual environment and dependencies

### Build Scripts

- **`rebuild_nixl_with_gds.sh`** - Comprehensive script to rebuild NIXL with GDS support
- **`rebuild_nixl_simple.sh`** - Simplified rebuild script for common scenarios

### Diagnostic Tools

- **`check_nixl_plugins.py`** - Diagnoses NIXL installation and available plugins
- **`nixl_gds_example_fallback.py`** - GDS example with fallback for missing GDS plugin


## Prerequisites

### 1. NVIDIA GDS (GPU Direct Storage)

NIXL requires NVIDIA GDS for optimal performance. Install it using the provided script:

```bash
# Make the script executable
chmod +x install_gds.sh

# Run the installation script
./install_gds.sh
```

If the main script fails, try the alternative installation script:

```bash
chmod +x install_gds_alt.sh
./install_gds_alt.sh
```

**Note**: The installation script will:
- Add the NVIDIA CUDA repository
- Update package lists
- Install the appropriate NVIDIA GDS package
- Clean up temporary files

### 2. Python Environment

Set up a Python virtual environment and install dependencies:

```bash
# Make the setup script executable
chmod +x setup_venv.sh

# Run the setup script
./setup_venv.sh

source venv/bin/activate

```

## Running the Benchmarks

### Basic NIXL Test

```bash
# Activate virtual environment
source venv/bin/activate

# Run the comprehensive NIXL test
python with_nxl_ucx.py
```

### Baseline Comparison

```bash
# Run baseline test without NIXL
python without_nixl.py
```

### GDS Example

### Rebuilding NIXL with GDS Support

If you need GDS functionality, you can rebuild NIXL with GDS support:


**Prerequisites for rebuilding:**
- NIXL source code (set `NIXL_SOURCE_URL` or place in `../nixl`, `../../nixl`, or `./nixl`)
  ` git clone https://github.com/ai-dynamo/nixl.git`
- CUDA installation (automatically detected)
- NVIDIA GDS (installed via `./install_gds.sh`)
- Build tools (automatically installed if missing)


```bash
# Navigate to the NIXL source directory


# Install in development mode to your virtual environment

cd ~/nixl_benchmark/nixl_build/nixl_source

sudo update-alternatives --install /usr/local/cuda cuda /usr/local/cuda-12.8 128

cd nixl_build/nixl_source && rm -rf build && export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH && meson setup build --prefix=/usr/local -Dgds_path=/usr/local/cuda-12.8

ninja -C build

sudo ninja -C build install
sudo ldconfig



```

**Verify GDS plugin availability:**

After installation, run the diagnostic script to confirm GDS support:

```bash
python check_nixl_plugins.py
```

You should now see `"GDS"` in the available plugins list.

## Dependencies

See `requirements.txt` for Python package dependencies:
- numpy
- torch
- nixl (NIXL Python bindings)


## test gds 

- First generate the test file 
```bash
python create_test_file.py
python nixl_gds_example.py test_file_5gb.dat
```

``


## License

This project is licensed under the Apache License, Version 2.0. See the license headers in individual files for details.

## Contributing

When contributing to this benchmark suite:
1. Ensure all prerequisites are documented
2. Test on multiple Ubuntu versions if possible
3. Update installation scripts for new dependencies
4. Add appropriate error handling and troubleshooting information 

