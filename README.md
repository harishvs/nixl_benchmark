# NIXL Benchmark Suite

This repository contains benchmark scripts and examples for testing NVIDIA's NIXL (NVIDIA Interconnect eXchange Library) framework, which provides high-performance data transfer capabilities for distributed computing environments.

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
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. NIXL Library

Ensure NIXL is properly installed and configured in your system. The `NIXL_PLUGIN_DIR` environment variable should be set to point to your NIXL plugins directory.

## Files Description

### Core Scripts

- **`with_nxl.py`** - Comprehensive NIXL benchmark demonstrating:
  - Agent setup and configuration
  - Memory registration and management
  - Data transfer operations (READ/WRITE)
  - Notification system
  - Resource cleanup

- **`without_nxl.py`** - Baseline benchmark without NIXL for comparison

- **`nixl_gds_example.py`** - Example demonstrating NIXL with GPU Direct Storage integration

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

## Running the Benchmarks

### Basic NIXL Test

```bash
# Activate virtual environment
source venv/bin/activate

# Run the comprehensive NIXL test
python with_nxl.py
```

### Baseline Comparison

```bash
# Run baseline test without NIXL
python without_nxl.py
```

### GDS Example

```bash
# Run NIXL with GDS example (requires GDS plugin)
python nixl_gds_example.py <file_path>

# Run GDS example with fallback (recommended)
python nixl_gds_example_fallback.py <file_path>
```

**Note**: The GDS example requires NVIDIA GDS to be installed and NIXL to be built with GDS support. If you encounter issues, use the fallback version or run the diagnostic script.

## Expected Output

The `with_nxl.py` script will output:
- NIXL plugin information and configuration
- Memory registration details
- Transfer operation status
- Completion confirmations for both initiator and target agents
- Resource cleanup confirmation

## Troubleshooting

### GDS Installation Issues

If you encounter issues installing NVIDIA GDS:

1. **Check Ubuntu version compatibility**: The script is configured for Ubuntu 22.04. For other versions, modify the repository URL in `install_gds.sh`.

2. **Manual repository addition**:
   ```bash
   # For Ubuntu 20.04
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
   
   # For Ubuntu 18.04
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.1-1_all.deb
   ```

3. **Check available packages**:
   ```bash
   apt search nvidia-gds
   ```

### NIXL Configuration Issues

- Ensure `NIXL_PLUGIN_DIR` environment variable is set correctly
- Verify NIXL library is properly installed
- Check that UCX backend is available

### GDS Plugin Issues

If the GDS example fails with "GDS not in plugin_list":

1. **Run diagnostic script**:
   ```bash
   python check_nixl_plugins.py
   ```

2. **Install NVIDIA GDS** (if not already installed):
   ```bash
   ./install_gds.sh
   ```

3. **Use fallback example**:
   ```bash
   python nixl_gds_example_fallback.py <file_path>
   ```

4. **Check NIXL build configuration**: Ensure NIXL was built with GDS support enabled

### Rebuilding NIXL with GDS Support

If you need GDS functionality, you can rebuild NIXL with GDS support:

```bash
# Simple rebuild (recommended)
./rebuild_nixl_simple.sh

# Comprehensive rebuild with detailed logging
./rebuild_nixl_with_gds.sh
```

**Prerequisites for rebuilding:**
- NIXL source code (set `NIXL_SOURCE_URL` or place in `../nixl`, `../../nixl`, or `./nixl`)
- CUDA installation (automatically detected)
- NVIDIA GDS (installed via `./install_gds.sh`)
- Build tools (automatically installed if missing)

**After rebuilding, update your Python environment:**

If you're using a virtual environment, you need to install the newly built NIXL:

```bash
# Navigate to the NIXL source directory
cd nixl_build/nixl_source

# Install in development mode to your virtual environment
pip install -e .
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

## License

This project is licensed under the Apache License, Version 2.0. See the license headers in individual files for details.

## Contributing

When contributing to this benchmark suite:
1. Ensure all prerequisites are documented
2. Test on multiple Ubuntu versions if possible
3. Update installation scripts for new dependencies
4. Add appropriate error handling and troubleshooting information 

## Final Troubleshooting and Environment Setup for NIXL + GDS

If you encounter issues with NIXL Python bindings or plugin detection, follow these steps to ensure your environment is correctly configured to use the system-installed NIXL with GDS support:

### 1. Remove `nixl` from `requirements.txt`
- Open `requirements.txt` and delete any line containing `nixl`.

### 2. Recreate Your Virtual Environment (if needed)
```bash
# Remove old venv if it exists
rm -rf venv

# Create a new virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements (without nixl)
pip install -r requirements.txt
pip install numpy  # If not already in requirements.txt
```

### 3. Set Environment Variables (every shell or add to ~/.bashrc)
```bash
export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export NIXL_PLUGIN_DIR=/usr/local/lib/x86_64-linux-gnu/plugins
export PYTHONPATH=/usr/local/lib/python3.12/site-packages:$PYTHONPATH
```
To make these changes permanent, add them to your `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export NIXL_PLUGIN_DIR=/usr/local/lib/x86_64-linux-gnu/plugins' >> ~/.bashrc
echo 'export PYTHONPATH=/usr/local/lib/python3.12/site-packages:$PYTHONPATH' >> ~/.bashrc
```

### 4. Test Your Setup
```bash
python3 -c "import nixl; print(nixl)"
python3 check_nixl_plugins.py
```
You should see all expected plugins, including `GDS`, in the output.

---

**Note:**
- Do **not** pip install `nixl` in your venv unless you are building from your own working source and know it will succeed.
- Always use the system install for NIXL when using custom builds with GDS support. 