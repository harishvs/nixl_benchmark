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
import time

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config

if __name__ == "__main__":
    # Increase buffer size for better performance with large files
    buf_size = 64 * 1024 * 1024  # 64 MB buffer for large file transfers
    # Allocate memory and register with NIXL

    if len(sys.argv) < 2:
        print("Please specify input file path in argv")
        exit(0)

    input_file = sys.argv[1]
    output_file = input_file + ".gds_output"
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("Using NIXL Plugins from:")
    print(os.environ["NIXL_PLUGIN_DIR"])

    agent_config = nixl_agent_config(backends=[])
    nixl_agent1 = nixl_agent("GDSTester", agent_config)

    plugin_list = nixl_agent1.get_plugin_list()
    assert "GDS" in plugin_list

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
    addr3 = nixl_utils.malloc_passthru(buf_size)  # Third buffer for input verification
    nixl_utils.ba_buf(addr1, buf_size)

    agent1_strings = [(addr1, buf_size, 0, "a"), (addr2, buf_size, 0, "b"), (addr3, buf_size, 0, "c")]

    agent1_reg_descs = nixl_agent1.get_reg_descs(agent1_strings, "DRAM")
    agent1_xfer1_descs = nixl_agent1.get_xfer_descs([(addr1, buf_size, 0)], "DRAM")
    agent1_xfer2_descs = nixl_agent1.get_xfer_descs([(addr2, buf_size, 0)], "DRAM")
    agent1_xfer3_descs = nixl_agent1.get_xfer_descs([(addr3, buf_size, 0)], "DRAM")

    assert nixl_agent1.register_memory(agent1_reg_descs) is not None

    # Open input file as read-only to preserve it
    agent1_fd_input = os.open(input_file, os.O_RDONLY)
    assert agent1_fd_input >= 0
    
    # Open output file for writing (will be created or truncated)
    agent1_fd_output = os.open(output_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    assert agent1_fd_output >= 0

    # Register input file for reading
    agent1_file_list_input = [(0, buf_size, agent1_fd_input, "b")]
    agent1_file_descs_input = nixl_agent1.register_memory(agent1_file_list_input, "FILE")
    assert agent1_file_descs_input is not None

    # Register output file for writing
    agent1_file_list_output = [(0, buf_size, agent1_fd_output, "b")]
    agent1_file_descs_output = nixl_agent1.register_memory(agent1_file_list_output, "FILE")
    assert agent1_file_descs_output is not None

    agent1_xfer_files_input = agent1_file_descs_input.trim()
    agent1_xfer_files_output = agent1_file_descs_output.trim()

    # First, read from input file into first buffer (GPU upload)
    print("Reading from input file to GPU...")
    start_time = time.time()
    xfer_handle_read = nixl_agent1.initialize_xfer(
        "READ", agent1_xfer1_descs, agent1_xfer_files_input, "GDSTester"
    )
    if not xfer_handle_read:
        print("Creating read transfer failed.")
        exit()

    state = nixl_agent1.transfer(xfer_handle_read)
    assert state != "ERR"

    done = False
    while not done:
        state = nixl_agent1.check_xfer_state(xfer_handle_read)
        if state == "ERR":
            print("Read transfer got to Error state.")
            exit()
        elif state == "DONE":
            done = True
            upload_time = time.time() - start_time
            upload_time_ms = upload_time * 1000
            print(f"GPU upload completed in {upload_time_ms:.2f} ms")
            print(f"Upload bandwidth: {buf_size / upload_time / (1024*1024):.2f} MB/s")

    # Write from first buffer to output file
    print("Writing to output file...")
    xfer_handle_write = nixl_agent1.initialize_xfer(
        "WRITE", agent1_xfer1_descs, agent1_xfer_files_output, "GDSTester"
    )
    if not xfer_handle_write:
        print("Creating write transfer failed.")
        exit()

    state = nixl_agent1.transfer(xfer_handle_write)
    assert state != "ERR"

    done = False
    while not done:
        state = nixl_agent1.check_xfer_state(xfer_handle_write)
        if state == "ERR":
            print("Write transfer got to Error state.")
            exit()
        elif state == "DONE":
            done = True
            print("Write transfer done")

    # Read output file data back into second buffer for verification (GPU download)
    print("Reading back from output file for verification (GPU download)...")
    start_time = time.time()
    xfer_handle_verify = nixl_agent1.initialize_xfer(
        "READ", agent1_xfer2_descs, agent1_xfer_files_output, "GDSTester"
    )
    if not xfer_handle_verify:
        print("Creating verification transfer failed.")
        exit()

    state = nixl_agent1.transfer(xfer_handle_verify)
    assert state != "ERR"

    done = False
    while not done:
        state = nixl_agent1.check_xfer_state(xfer_handle_verify)
        if state == "ERR":
            print("Verification transfer got to Error state.")
            exit()
        elif state == "DONE":
            done = True
            download_time = time.time() - start_time
            download_time_ms = download_time * 1000
            print(f"GPU download completed in {download_time_ms:.2f} ms")
            print(f"Download bandwidth: {buf_size / download_time / (1024*1024):.2f} MB/s")

    # Read original input file into third buffer for comparison
    print("Reading original input file for comparison...")
    xfer_handle_input_verify = nixl_agent1.initialize_xfer(
        "READ", agent1_xfer3_descs, agent1_xfer_files_input, "GDSTester"
    )
    if not xfer_handle_input_verify:
        print("Creating input verification transfer failed.")
        exit()

    state = nixl_agent1.transfer(xfer_handle_input_verify)
    assert state != "ERR"

    done = False
    while not done:
        state = nixl_agent1.check_xfer_state(xfer_handle_input_verify)
        if state == "ERR":
            print("Input verification transfer got to Error state.")
            exit()
        elif state == "DONE":
            done = True
            print("Input verification transfer done")

    # Verification: Compare original input (addr3) with output file data (addr2)
    print("Verifying input file matches output file...")
    nixl_utils.verify_transfer(addr3, addr2, buf_size)
    print("✓ Input file and output file match!")

    # Cleanup
    nixl_agent1.release_xfer_handle(xfer_handle_read)
    nixl_agent1.release_xfer_handle(xfer_handle_write)
    nixl_agent1.release_xfer_handle(xfer_handle_verify)
    nixl_agent1.release_xfer_handle(xfer_handle_input_verify)
    nixl_agent1.deregister_memory(agent1_reg_descs)
    nixl_agent1.deregister_memory(agent1_file_descs_input)
    nixl_agent1.deregister_memory(agent1_file_descs_output)

    nixl_utils.free_passthru(addr1)
    nixl_utils.free_passthru(addr2)
    nixl_utils.free_passthru(addr3)

    os.close(agent1_fd_input)
    os.close(agent1_fd_output)

    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Buffer size: {buf_size / (1024*1024):.2f} MB")
    print(f"GPU Upload time: {upload_time_ms:.2f} ms")
    print(f"GPU Upload bandwidth: {buf_size / upload_time / (1024*1024):.2f} MB/s")
    print(f"GPU Download time: {download_time_ms:.2f} ms")
    print(f"GPU Download bandwidth: {buf_size / download_time / (1024*1024):.2f} MB/s")
    print(f"Total transfer time: {(upload_time + download_time) * 1000:.2f} ms")
    print(f"Average bandwidth: {2 * buf_size / (upload_time + download_time) / (1024*1024):.2f} MB/s")
    print("="*50)
    print("Test Complete.")
    print(f"Input file '{input_file}' was preserved.")
    print(f"Output written to '{output_file}'.")
