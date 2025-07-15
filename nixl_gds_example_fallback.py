#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config

def run_gds_example():
    """Run GDS example with fallback to available plugins"""
    
    buf_size = 16 * 4096
    
    if len(sys.argv) < 2:
        print("Please specify file path in argv")
        print("Usage: python nixl_gds_example_fallback.py <file_path>")
        exit(1)

    print("Using NIXL Plugins from:")
    print(os.environ["NIXL_PLUGIN_DIR"])

    agent_config = nixl_agent_config(backends=[])
    nixl_agent1 = nixl_agent("GDSTester", agent_config)

    plugin_list = nixl_agent1.get_plugin_list()
    print(f"Available plugins: {plugin_list}")
    
    # Check if GDS is available
    if "GDS" not in plugin_list:
        print("\n⚠ GDS plugin not available!")
        print("This example requires NVIDIA GDS to be installed and NIXL to be built with GDS support.")
        print("\nTo install NVIDIA GDS:")
        print("  ./install_gds.sh")
        print("\nTo check your NIXL installation:")
        print("  python check_nixl_plugins.py")
        print("\nAvailable plugins for testing:")
        
        # Show available plugins and their capabilities
        for plugin in plugin_list:
            print(f"\nPlugin: {plugin}")
            try:
                mem_types = nixl_agent1.get_plugin_mem_types(plugin)
                print(f"  Memory types: {mem_types}")
            except Exception as e:
                print(f"  Error getting memory types: {e}")
        
        print(f"\nWould you like to run a basic test with available plugins instead? (y/n): ", end="")
        response = input().lower().strip()
        
        if response != 'y':
            print("Exiting. Please install GDS and try again.")
            exit(1)
        
        # Run basic test with available plugins
        run_basic_test(nixl_agent1, plugin_list[0] if plugin_list else None)
        return

    print("✓ GDS plugin found!")
    
    # Continue with original GDS example
    print("Plugin parameters")
    print(nixl_agent1.get_plugin_mem_types("GDS"))
    print(nixl_agent1.get_plugin_params("GDS"))

    nixl_agent1.create_backend("GDS")

    print("\nLoaded backend parameters")
    print(nixl_agent1.get_backend_mem_types("GDS"))
    print(nixl_agent1.get_backend_params("GDS"))
    print()

    # get DRAM buf and initialize it to 0xba for verification
    addr1 = nixl_utils.malloc_passthru(buf_size)
    addr2 = nixl_utils.malloc_passthru(buf_size)
    nixl_utils.ba_buf(addr1, buf_size)

    agent1_strings = [(addr1, buf_size, 0, "a"), (addr2, buf_size, 0, "b")]

    agent1_reg_descs = nixl_agent1.get_reg_descs(agent1_strings, "DRAM")
    agent1_xfer1_descs = nixl_agent1.get_xfer_descs([(addr1, buf_size, 0)], "DRAM")
    agent1_xfer2_descs = nixl_agent1.get_xfer_descs([(addr2, buf_size, 0)], "DRAM")

    assert nixl_agent1.register_memory(agent1_reg_descs) is not None

    # user must pass full file path for test
    agent1_fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)
    assert agent1_fd >= 0

    agent1_file_list = [(0, buf_size, agent1_fd, "b")]

    agent1_file_descs = nixl_agent1.register_memory(agent1_file_list, "FILE")
    assert agent1_file_descs is not None

    agent1_xfer_files = agent1_file_descs.trim()

    # initialize transfer mode
    xfer_handle_1 = nixl_agent1.initialize_xfer(
        "WRITE", agent1_xfer1_descs, agent1_xfer_files, "GDSTester"
    )
    if not xfer_handle_1:
        print("Creating transfer failed.")
        exit()

    state = nixl_agent1.transfer(xfer_handle_1)
    assert state != "ERR"

    done = False

    while not done:
        state = nixl_agent1.check_xfer_state(xfer_handle_1)
        if state == "ERR":
            print("Transfer got to Error state.")
            exit()
        elif state == "DONE":
            done = True
            print("Initiator done")

    # read file data back into second buffer
    xfer_handle_2 = nixl_agent1.initialize_xfer(
        "READ", agent1_xfer2_descs, agent1_xfer_files, "GDSTester"
    )
    if not xfer_handle_2:
        print("Creating transfer failed.")
        exit()

    state = nixl_agent1.transfer(xfer_handle_2)
    assert state != "ERR"

    done = False

    while not done:
        state = nixl_agent1.check_xfer_state(xfer_handle_2)
        if state == "ERR":
            print("Transfer got to Error state.")
            exit()
        elif state == "DONE":
            done = True
            print("Initiator done")

    # transfer verification
    nixl_utils.verify_transfer(addr1, addr2, buf_size)

    nixl_agent1.release_xfer_handle(xfer_handle_1)
    nixl_agent1.release_xfer_handle(xfer_handle_2)
    nixl_agent1.deregister_memory(agent1_reg_descs)
    nixl_agent1.deregister_memory(agent1_file_descs)

    nixl_utils.free_passthru(addr1)
    nixl_utils.free_passthru(addr2)

    os.close(agent1_fd)

    print("GDS Test Complete.")

def run_basic_test(agent, plugin_name):
    """Run a basic test with available plugins"""
    
    if not plugin_name:
        print("No plugins available for testing.")
        return
    
    print(f"\nRunning basic test with plugin: {plugin_name}")
    
    try:
        # Create backend
        agent.create_backend(plugin_name)
        print(f"✓ Created backend: {plugin_name}")
        
        # Get backend info
        mem_types = agent.get_backend_mem_types(plugin_name)
        params = agent.get_backend_params(plugin_name)
        print(f"Memory types: {mem_types}")
        print(f"Parameters: {params}")
        
        print("✓ Basic plugin test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error testing plugin {plugin_name}: {e}")

if __name__ == "__main__":
    run_gds_example() 