#!/usr/bin/env python3
"""
Linux-based Memory Experiments
Demonstrates actual page faults, memory allocation, and system behavior
"""

import os
import sys
import time
import mmap
import subprocess
import resource
import gc
from typing import List, Dict
import psutil

class LinuxMemoryExperiment:
    """Class for conducting real Linux memory experiments"""
    
    def __init__(self):
        self.pid = os.getpid()
        self.page_size = os.sysconf(os.sysconf_names['SC_PAGE_SIZE'])
        print(f"Process PID: {self.pid}")
        print(f"System Page Size: {self.page_size} bytes")
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics for the process"""
        try:
            # Get process memory info
            process = psutil.Process(self.pid)
            memory_info = process.memory_info()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            # Get resource usage (includes page faults)
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            
            stats = {
                'rss': memory_info.rss,  # Resident Set Size
                'vms': memory_info.vms,  # Virtual Memory Size
                'major_page_faults': rusage.ru_majflt,  # Major page faults
                'minor_page_faults': rusage.ru_minflt,  # Minor page faults
                'system_available': system_memory.available,
                'system_used_percent': system_memory.percent
            }
            
            return stats
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}
    
    def print_memory_stats(self, label: str, stats: Dict):
        """Print formatted memory statistics"""
        print(f"\n=== {label} ===")
        print(f"RSS (Physical Memory): {stats.get('rss', 0) / 1024 / 1024:.2f} MB")
        print(f"VMS (Virtual Memory): {stats.get('vms', 0) / 1024 / 1024:.2f} MB")
        print(f"Major Page Faults: {stats.get('major_page_faults', 0)}")
        print(f"Minor Page Faults: {stats.get('minor_page_faults', 0)}")
        print(f"System Memory Available: {stats.get('system_available', 0) / 1024 / 1024:.2f} MB")
        print(f"System Memory Used: {stats.get('system_used_percent', 0):.1f}%")
    
    def experiment_1_basic_allocation(self):
        """Experiment 1: Basic memory allocation and page faults"""
        print("\n" + "="*60)
        print("EXPERIMENT 1: Basic Memory Allocation")
        print("="*60)
        
        # Get initial stats
        initial_stats = self.get_memory_stats()
        self.print_memory_stats("Initial State", initial_stats)
        
        # Allocate memory in chunks
        chunk_size = 10 * 1024 * 1024  # 10 MB chunks
        chunks = []
        
        for i in range(5):
            print(f"\nAllocating chunk {i+1} ({chunk_size // 1024 // 1024} MB)...")
            
            # Allocate but don't touch the memory yet
            chunk = bytearray(chunk_size)
            chunks.append(chunk)
            
            stats = self.get_memory_stats()
            print(f"  VMS: {stats['vms'] / 1024 / 1024:.2f} MB, "
                  f"RSS: {stats['rss'] / 1024 / 1024:.2f} MB")
        
        print("\nNow touching allocated memory (forcing page faults)...")
        
        for i, chunk in enumerate(chunks):
            print(f"Touching chunk {i+1}...")
            # Write to every page to force page faults
            for offset in range(0, len(chunk), self.page_size):
                chunk[offset] = 0xFF
            
            stats = self.get_memory_stats()
            print(f"  After touch {i+1} - RSS: {stats['rss'] / 1024 / 1024:.2f} MB, "
                  f"Minor faults: {stats['minor_page_faults']}")
        
        final_stats = self.get_memory_stats()
        self.print_memory_stats("Final State", final_stats)
        
        # Calculate differences
        print("\n=== Changes ===")
        print(f"RSS increase: {(final_stats['rss'] - initial_stats['rss']) / 1024 / 1024:.2f} MB")
        print(f"VMS increase: {(final_stats['vms'] - initial_stats['vms']) / 1024 / 1024:.2f} MB")
        print(f"Minor page faults: {final_stats['minor_page_faults'] - initial_stats['minor_page_faults']}")
        print(f"Major page faults: {final_stats['major_page_faults'] - initial_stats['major_page_faults']}")
        
        # Clean up
        del chunks
        gc.collect()
    
    def experiment_2_memory_mapping(self):
        """Experiment 2: Memory-mapped files and page faults"""
        print("\n" + "="*60)
        print("EXPERIMENT 2: Memory-Mapped Files")
        print("="*60)
        
        # Create a temporary file
        temp_file = "/tmp/memory_test.dat"
        file_size = 50 * 1024 * 1024  # 50 MB
        
        print(f"Creating {file_size // 1024 // 1024} MB temporary file...")
        
        # Create and write to file
        with open(temp_file, "wb") as f:
            f.write(b"A" * file_size)
        
        initial_stats = self.get_memory_stats()
        self.print_memory_stats("Before Memory Mapping", initial_stats)
        
        # Memory map the file
        print("\nMemory mapping the file...")
        with open(temp_file, "r+b") as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                after_map_stats = self.get_memory_stats()
                print(f"After mapping - VMS: {after_map_stats['vms'] / 1024 / 1024:.2f} MB")
                
                # Read from different parts of the file (causing page faults)
                print("\nReading from mapped memory...")
                positions = [0, file_size//4, file_size//2, 3*file_size//4, file_size-1]
                
                for i, pos in enumerate(positions):
                    print(f"Reading position {pos} ({pos // 1024 // 1024} MB offset)...")
                    data = mm[pos:pos+1]  # Read one byte
                    
                    stats = self.get_memory_stats()
                    print(f"  RSS: {stats['rss'] / 1024 / 1024:.2f} MB, "
                          f"Minor faults: {stats['minor_page_faults']}")
                
                # Sequential read to see page fault pattern
                print("\nSequential read (triggering more page faults)...")
                start_stats = self.get_memory_stats()
                
                # Read every 1MB
                for offset in range(0, file_size, 1024*1024):
                    mm[offset:offset+1]
                
                end_stats = self.get_memory_stats()
                print(f"Sequential read caused {end_stats['minor_page_faults'] - start_stats['minor_page_faults']} minor page faults")
        
        # Clean up
        os.unlink(temp_file)
        
        final_stats = self.get_memory_stats()
        self.print_memory_stats("After Cleanup", final_stats)
    
    def experiment_3_memory_pressure(self):
        """Experiment 3: Memory pressure and swapping"""
        print("\n" + "="*60)
        print("EXPERIMENT 3: Memory Pressure Simulation")
        print("="*60)
        
        # Get available memory
        system_memory = psutil.virtual_memory()
        print(f"System total memory: {system_memory.total / 1024 / 1024:.2f} MB")
        print(f"System available memory: {system_memory.available / 1024 / 1024:.2f} MB")
        
        # Allocate a significant portion of available memory
        # (Be careful not to cause system instability)
        target_allocation = min(system_memory.available // 4, 500 * 1024 * 1024)  # Max 500MB
        
        print(f"Target allocation: {target_allocation / 1024 / 1024:.2f} MB")
        
        initial_stats = self.get_memory_stats()
        self.print_memory_stats("Initial State", initial_stats)
        
        # Allocate memory progressively
        chunks = []
        chunk_size = 50 * 1024 * 1024  # 50 MB chunks
        
        allocated = 0
        while allocated < target_allocation:
            current_chunk_size = min(chunk_size, target_allocation - allocated)
            
            print(f"\nAllocating {current_chunk_size / 1024 / 1024:.2f} MB...")
            
            # Allocate and immediately touch the memory
            chunk = bytearray(current_chunk_size)
            for i in range(0, len(chunk), self.page_size):
                chunk[i] = 0xAA
            
            chunks.append(chunk)
            allocated += current_chunk_size
            
            stats = self.get_memory_stats()
            system_stats = psutil.virtual_memory()
            
            print(f"  Process RSS: {stats['rss'] / 1024 / 1024:.2f} MB")
            print(f"  System available: {system_stats.available / 1024 / 1024:.2f} MB")
            print(f"  Minor page faults: {stats['minor_page_faults']}")
            print(f"  Major page faults: {stats['major_page_faults']}")
            
            # Small delay to observe system behavior
            time.sleep(0.5)
        
        print("\nMemory allocation complete. Testing access patterns...")
        
        # Random access pattern to test page replacement
        import random
        for i in range(10):
            chunk_idx = random.randint(0, len(chunks) - 1)
            offset = random.randint(0, len(chunks[chunk_idx]) - 1)
            chunks[chunk_idx][offset] = 0xBB
            
            if i % 3 == 0:  # Print stats every few accesses
                stats = self.get_memory_stats()
                print(f"Random access {i+1}: Minor faults = {stats['minor_page_faults']}")
        
        final_stats = self.get_memory_stats()
        self.print_memory_stats("Final State", final_stats)
        
        # Cleanup
        del chunks
        gc.collect()
        
        after_cleanup_stats = self.get_memory_stats()
        print(f"\nAfter cleanup RSS: {after_cleanup_stats['rss'] / 1024 / 1024:.2f} MB")
    
    def experiment_4_page_fault_analysis(self):
        """Experiment 4: Detailed page fault analysis"""
        print("\n" + "="*60)
        print("EXPERIMENT 4: Page Fault Pattern Analysis")
        print("="*60)
        
        # Different access patterns
        patterns = [
            ("Sequential", lambda arr, size: [arr[i] for i in range(0, size, self.page_size)]),
            ("Random", lambda arr, size: [arr[random.randint(0, size-1)] for _ in range(size//self.page_size)]),
            ("Strided", lambda arr, size: [arr[i] for i in range(0, size, self.page_size * 2)]),
        ]
        
        array_size = 20 * 1024 * 1024  # 20 MB
        
        for pattern_name, access_func in patterns:
            print(f"\n--- {pattern_name} Access Pattern ---")
            
            # Create array
            test_array = bytearray(array_size)
            
            # Get initial stats
            initial_stats = self.get_memory_stats()
            
            #
