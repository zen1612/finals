#!/usr/bin/env python3
"""
Virtual Memory Simulator
Implements page replacement algorithms, TLB simulation, and address translation
"""

import random
from collections import OrderedDict, deque
from typing import List, Dict, Tuple, Optional
import time
import matplotlib.pyplot as plt
import numpy as np

class TLB:
    """Translation Lookaside Buffer implementation"""
    
    def __init__(self, size: int = 4):
        self.size = size
        self.cache = OrderedDict()  # LRU ordering
        self.hits = 0
        self.misses = 0
    
    def lookup(self, page_num: int) -> Optional[int]:
        """Look up page number in TLB, return frame number if hit"""
        if page_num in self.cache:
            # Move to end (most recently used)
            frame_num = self.cache.pop(page_num)
            self.cache[page_num] = frame_num
            self.hits += 1
            return frame_num
        else:
            self.misses += 1
            return None
    
    def update(self, page_num: int, frame_num: int):
        """Update TLB with new page->frame mapping"""
        if page_num in self.cache:
            self.cache.pop(page_num)
        elif len(self.cache) >= self.size:
            # Remove least recently used
            self.cache.popitem(last=False)
        
        self.cache[page_num] = frame_num
    
    def get_hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset_stats(self):
        self.hits = 0
        self.misses = 0
    
    def __str__(self) -> str:
        return f"TLB Contents: {dict(self.cache)}"

class PageTable:
    """Page Table implementation"""
    
    def __init__(self):
        self.table = {}  # page_num -> frame_num
        self.valid_bits = {}  # page_num -> bool
    
    def get_frame(self, page_num: int) -> Optional[int]:
        """Get frame number for page, return None if not in memory"""
        if page_num in self.table and self.valid_bits.get(page_num, False):
            return self.table[page_num]
        return None
    
    def set_mapping(self, page_num: int, frame_num: int):
        """Set page to frame mapping"""
        self.table[page_num] = frame_num
        self.valid_bits[page_num] = True
    
    def invalidate_page(self, page_num: int):
        """Mark page as not in memory"""
        self.valid_bits[page_num] = False
    
    def get_page_for_frame(self, frame_num: int) -> Optional[int]:
        """Get page number currently in given frame"""
        for page_num, f_num in self.table.items():
            if f_num == frame_num and self.valid_bits.get(page_num, False):
                return page_num
        return None

class PageReplacementAlgorithm:
    """Base class for page replacement algorithms"""
    
    def __init__(self, num_frames: int):
        self.num_frames = num_frames
        self.frames = [-1] * num_frames  # -1 indicates empty frame
        self.page_faults = 0
    
    def access_page(self, page_num: int) -> Tuple[bool, Optional[int]]:
        """
        Access a page, return (fault_occurred, evicted_page)
        """
        raise NotImplementedError
    
    def is_full(self) -> bool:
        return -1 not in self.frames
    
    def find_empty_frame(self) -> Optional[int]:
        try:
            return self.frames.index(-1)
        except ValueError:
            return None
    
    def reset(self):
        self.frames = [-1] * self.num_frames
        self.page_faults = 0

class FIFOPageReplacement(PageReplacementAlgorithm):
    """First-In-First-Out page replacement"""
    
    def __init__(self, num_frames: int):
        super().__init__(num_frames)
        self.queue = deque()
    
    def access_page(self, page_num: int) -> Tuple[bool, Optional[int]]:
        # Check if page is already in memory
        if page_num in self.frames:
            return False, None  # No fault
        
        # Page fault occurred
        self.page_faults += 1
        evicted_page = None
        
        if not self.is_full():
            # Find empty frame
            frame_idx = self.find_empty_frame()
            self.frames[frame_idx] = page_num
            self.queue.append(frame_idx)
        else:
            # Replace oldest page
            frame_idx = self.queue.popleft()
            evicted_page = self.frames[frame_idx]
            self.frames[frame_idx] = page_num
            self.queue.append(frame_idx)
        
        return True, evicted_page
    
    def reset(self):
        super().reset()
        self.queue.clear()

class LRUPageReplacement(PageReplacementAlgorithm):
    """Least Recently Used page replacement"""
    
    def __init__(self, num_frames: int):
        super().__init__(num_frames)
        self.access_order = OrderedDict()  # page_num -> frame_idx
    
    def access_page(self, page_num: int) -> Tuple[bool, Optional[int]]:
        # Check if page is already in memory
        if page_num in self.access_order:
            # Update access order (move to end)
            frame_idx = self.access_order.pop(page_num)
            self.access_order[page_num] = frame_idx
            return False, None  # No fault
        
        # Page fault occurred
        self.page_faults += 1
        evicted_page = None
        
        if not self.is_full():
            # Find empty frame
            frame_idx = self.find_empty_frame()
            self.frames[frame_idx] = page_num
            self.access_order[page_num] = frame_idx
        else:
            # Replace least recently used page
            lru_page, frame_idx = self.access_order.popitem(last=False)
            evicted_page = self.frames[frame_idx]
            self.frames[frame_idx] = page_num
            self.access_order[page_num] = frame_idx
        
        return True, evicted_page
    
    def reset(self):
        super().reset()
        self.access_order.clear()

class OptimalPageReplacement(PageReplacementAlgorithm):
    """Optimal page replacement (requires future knowledge)"""
    
    def __init__(self, num_frames: int, reference_string: List[int]):
        super().__init__(num_frames)
        self.reference_string = reference_string
        self.current_pos = 0
    
    def access_page(self, page_num: int) -> Tuple[bool, Optional[int]]:
        # Check if page is already in memory
        if page_num in self.frames:
            self.current_pos += 1
            return False, None  # No fault
        
        # Page fault occurred
        self.page_faults += 1
        evicted_page = None
        
        if not self.is_full():
            # Find empty frame
            frame_idx = self.find_empty_frame()
            self.frames[frame_idx] = page_num
        else:
            # Find page that will be used farthest in the future
            farthest_use = {}
            for i, frame_page in enumerate(self.frames):
                if frame_page == -1:
                    continue
                
                # Find next use of this page
                next_use = float('inf')
                for j in range(self.current_pos + 1, len(self.reference_string)):
                    if self.reference_string[j] == frame_page:
                        next_use = j
                        break
                farthest_use[i] = next_use
            
            # Replace page with farthest next use
            frame_idx = max(farthest_use.keys(), key=lambda x: farthest_use[x])
            evicted_page = self.frames[frame_idx]
            self.frames[frame_idx] = page_num
        
        self.current_pos += 1
        return True, evicted_page
    
    def reset(self):
        super().reset()
        self.current_pos = 0

class AdaptivePageReplacement(PageReplacementAlgorithm):
    """
    Custom adaptive algorithm that combines LRU with frequency tracking
    Uses both recency and frequency to make replacement decisions
    """
    
    def __init__(self, num_frames: int, window_size: int = 10):
        super().__init__(num_frames)
        self.access_order = OrderedDict()  # page_num -> (frame_idx, access_count)
        self.window_size = window_size
        self.recent_accesses = deque(maxlen=window_size)
    
    def access_page(self, page_num: int) -> Tuple[bool, Optional[int]]:
        self.recent_accesses.append(page_num)
        
        # Check if page is already in memory
        if page_num in self.access_order:
            # Update access order and frequency
            frame_idx, count = self.access_order.pop(page_num)
            self.access_order[page_num] = (frame_idx, count + 1)
            return False, None  # No fault
        
        # Page fault occurred
        self.page_faults += 1
        evicted_page = None
        
        if not self.is_full():
            # Find empty frame
            frame_idx = self.find_empty_frame()
            self.frames[frame_idx] = page_num
            self.access_order[page_num] = (frame_idx, 1)
        else:
            # Calculate replacement score (lower is worse)
            # Score = frequency_weight * frequency + recency_weight * (1/position_from_end)
            scores = {}
            recent_freq = {}
            
            # Count frequency in recent window
            for page in self.recent_accesses:
                recent_freq[page] = recent_freq.get(page, 0) + 1
            
            for page_num_in_mem, (frame_idx, total_count) in self.access_order.items():
                recent_count = recent_freq.get(page_num_in_mem, 0)
                # Combine recent frequency with total frequency
                frequency_score = 0.7 * recent_count + 0.3 * total_count
                
                # Recency score (higher for more recent)
                pages_list = list(self.access_order.keys())
                recency_score = len(pages_list) - pages_list.index(page_num_in_mem)
                
                # Combined score (higher is better)
                scores[page_num_in_mem] = 0.6 * frequency_score + 0.4 * recency_score
            
            # Replace page with lowest score
            victim_page = min(scores.keys(), key=lambda x: scores[x])
            frame_idx, _ = self.access_order.pop(victim_page)
            evicted_page = self.frames[frame_idx]
            self.frames[frame_idx] = page_num
            self.access_order[page_num] = (frame_idx, 1)
        
        return True, evicted_page
    
    def reset(self):
        super().reset()
        self.access_order.clear()
        self.recent_accesses.clear()

class VirtualMemorySimulator:
    """Main virtual memory simulator"""
    
    def __init__(self, page_size: int = 4096, tlb_size: int = 4, num_frames: int = 3):
        self.page_size = page_size
        self.num_frames = num_frames
        self.tlb = TLB(tlb_size)
        self.page_table = PageTable()
        self.algorithms = {}
        self.current_algorithm = None
        
        # Statistics
        self.total_accesses = 0
        self.page_fault_history = []
        self.tlb_hit_history = []
    
    def add_algorithm(self, name: str, algorithm: PageReplacementAlgorithm):
        """Add a page replacement algorithm"""
        self.algorithms[name] = algorithm
    
    def set_algorithm(self, name: str):
        """Set the current page replacement algorithm"""
        if name in self.algorithms:
            self.current_algorithm = self.algorithms[name]
        else:
            raise ValueError(f"Algorithm '{name}' not found")
    
    def virtual_to_physical(self, virtual_addr: int) -> Tuple[int, int]:
        """Convert virtual address to page number and offset"""
        page_num = virtual_addr // self.page_size
        offset = virtual_addr % self.page_size
        return page_num, offset
    
    def access_memory(self, virtual_addr: int) -> Dict:
        """Access memory at virtual address, return access information"""
        self.total_accesses += 1
        page_num, offset = self.virtual_to_physical(virtual_addr)
        
        result = {
            'virtual_addr': virtual_addr,
            'page_num': page_num,
            'offset': offset,
            'tlb_hit': False,
            'page_fault': False,
            'physical_addr': None,
            'evicted_page': None
        }
        
        # Check TLB first
        frame_num = self.tlb.lookup(page_num)
        if frame_num is not None:
            result['tlb_hit'] = True
            result['physical_addr'] = frame_num * self.page_size + offset
            self.tlb_hit_history.append(1)
            return result
        
        self.tlb_hit_history.append(0)
        
        # TLB miss, check page table
        frame_num = self.page_table.get_frame(page_num)
        if frame_num is not None:
            # Page in memory, update TLB
            self.tlb.update(page_num, frame_num)
            result['physical_addr'] = frame_num * self.page_size + offset
            return result
        
        # Page fault - use current algorithm
        if self.current_algorithm is None:
            raise ValueError("No page replacement algorithm set")
        
        fault_occurred, evicted_page = self.current_algorithm.access_page(page_num)
        result['page_fault'] = fault_occurred
        result['evicted_page'] = evicted_page
        
        if fault_occurred:
            # Find which frame the page was loaded into
            frame_num = None
            for i, frame_page in enumerate(self.current_algorithm.frames):
                if frame_page == page_num:
                    frame_num = i
                    break
            
            # Update page table
            if evicted_page is not None:
                self.page_table.invalidate_page(evicted_page)
            self.page_table.set_mapping(page_num, frame_num)
            
            # Update TLB
            self.tlb.update(page_num, frame_num)
            
            result['physical_addr'] = frame_num * self.page_size + offset
        
        self.page_fault_history.append(1 if fault_occurred else 0)
        return result
    
    def simulate_reference_string(self, reference_string: List[int]) -> List[Dict]:
        """Simulate a complete reference string"""
        results = []
        for addr in reference_string:
            result = self.access_memory(addr)
            results.append(result)
        return results
    
    def reset(self):
        """Reset simulator state"""
        self.tlb.reset_stats()
        self.total_accesses = 0
        self.page_fault_history.clear()
        self.tlb_hit_history.clear()
        for algorithm in self.algorithms.values():
            algorithm.reset()
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        page_faults = sum(self.page_fault_history)
        tlb_hits = sum(self.tlb_hit_history)
        
        return {
            'total_accesses': self.total_accesses,
            'page_faults': page_faults,
            'page_fault_rate': page_faults / self.total_accesses if self.total_accesses > 0 else 0,
            'tlb_hits': tlb_hits,
            'tlb_misses': len(self.tlb_hit_history) - tlb_hits,
            'tlb_hit_ratio': self.tlb.get_hit_ratio(),
            'algorithm_page_faults': self.current_algorithm.page_faults if self.current_algorithm else 0
        }
    
    def visualize_memory_state(self):
        """Print current memory state"""
        print(f"\n=== Memory State ===")
        print(f"Physical Frames: {self.current_algorithm.frames if self.current_algorithm else 'N/A'}")
        print(f"TLB: {self.tlb}")
        print(f"Page Table Valid Pages: {[p for p, v in self.page_table.valid_bits.items() if v]}")

def generate_random_reference_string(length: int, page_range: int, locality_factor: float = 0.7) -> List[int]:
    """Generate a reference string with some locality of reference"""
    reference_string = []
    current_page = random.randint(0, page_range - 1)
    
    for _ in range(length):
        if random.random() < locality_factor:
            # Stay in locality (within ±2 pages)
            offset = random.choice([-2, -1, 0, 1, 2])
            current_page = max(0, min(page_range - 1, current_page + offset))
        else:
            # Jump to random page
            current_page = random.randint(0, page_range - 1)
        
        # Convert page number to virtual address
        virtual_addr = current_page * 4096 + random.randint(0, 4095)
        reference_string.append(virtual_addr)
    
    return reference_string

def compare_algorithms():
    """Compare different page replacement algorithms"""
    print("=== Virtual Memory Simulator ===\n")
    
    # Generate test reference string
    reference_string = generate_random_reference_string(50, 8, 0.6)
    page_nums = [addr // 4096 for addr in reference_string]
    print(f"Reference string (page numbers): {page_nums[:20]}{'...' if len(page_nums) > 20 else ''}")
    
    # Initialize simulator
    simulator = VirtualMemorySimulator(page_size=4096, tlb_size=4, num_frames=3)
    
    # Add algorithms
    simulator.add_algorithm("FIFO", FIFOPageReplacement(3))
    simulator.add_algorithm("LRU", LRUPageReplacement(3))
    simulator.add_algorithm("Optimal", OptimalPageReplacement(3, page_nums))
    simulator.add_algorithm("Adaptive", AdaptivePageReplacement(3, window_size=8))
    
    results = {}
    
    # Test each algorithm
    for alg_name in ["FIFO", "LRU", "Adaptive", "Optimal"]:
        print(f"\n--- Testing {alg_name} Algorithm ---")
        simulator.reset()
        simulator.set_algorithm(alg_name)
        
        # Simulate reference string
        access_results = simulator.simulate_reference_string(reference_string)
        stats = simulator.get_statistics()
        results[alg_name] = stats
        
        # Print results
        print(f"Page Faults: {stats['page_faults']}")
        print(f"Page Fault Rate: {stats['page_fault_rate']:.3f}")
        print(f"TLB Hit Ratio: {stats['tlb_hit_ratio']:.3f}")
        print(f"Total Memory Accesses: {stats['total_accesses']}")
        
        # Show first few accesses for demonstration
        print("\nFirst 10 memory accesses:")
        for i, result in enumerate(access_results[:10]):
            status = []
            if result['tlb_hit']:
                status.append("TLB HIT")
            if result['page_fault']:
                status.append("PAGE FAULT")
            if result['evicted_page'] is not None:
                status.append(f"EVICTED: {result['evicted_page']}")
            
            status_str = " | ".join(status) if status else "HIT"
            print(f"  {i+1:2d}. Addr:{result['virtual_addr']:5d} Page:{result['page_num']:2d} -> "
                  f"Phys:{result['physical_addr'] or 'N/A':8s} [{status_str}]")
        
        simulator.visualize_memory_state()
    
    # Summary comparison
    print(f"\n=== Algorithm Comparison Summary ===")
    print(f"{'Algorithm':<10} {'Page Faults':<12} {'Fault Rate':<12} {'TLB Hit Rate':<12}")
    print("-" * 50)
    for alg_name, stats in results.items():
        print(f"{alg_name:<10} {stats['page_faults']:<12} "
              f"{stats['page_fault_rate']:<12.3f} {stats['tlb_hit_ratio']:<12.3f}")

def interactive_demo():
    """Interactive demonstration"""
    print("\n=== Interactive Virtual Memory Demo ===")
    simulator = VirtualMemorySimulator(page_size=4096, tlb_size=4, num_frames=3)
    simulator.add_algorithm("LRU", LRUPageReplacement(3))
    simulator.set_algorithm("LRU")
    
    print("Enter virtual addresses to access (or 'quit' to exit)")
    print("Page size: 4096 bytes, Physical frames: 3, TLB size: 4")
    
    while True:
        try:
            user_input = input("\nEnter virtual address: ").strip()
            if user_input.lower() == 'quit':
                break
            
            virtual_addr = int(user_input)
            result = simulator.access_memory(virtual_addr)
            
            print(f"Virtual Address: {result['virtual_addr']}")
            print(f"Page Number: {result['page_num']}, Offset: {result['offset']}")
            print(f"Physical Address: {result['physical_addr']}")
            print(f"TLB Hit: {'Yes' if result['tlb_hit'] else 'No'}")
            print(f"Page Fault: {'Yes' if result['page_fault'] else 'No'}")
            if result['evicted_page'] is not None:
                print(f"Evicted Page: {result['evicted_page']}")
            
            simulator.visualize_memory_state()
            
        except ValueError:
            print("Please enter a valid integer address")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Run comparison
    compare_algorithms()
    
    # Uncomment for interactive demo
    # interactive_demo()
    
    print("\n=== Additional Test Cases ===")
    
    # Test with known reference string
    test_pages = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    test_addresses = [page * 4096 for page in test_pages]
    
    print(f"Test reference string: {test_pages}")
    
    simulator = VirtualMemorySimulator()
    simulator.add_algorithm("FIFO", FIFOPageReplacement(3))
    simulator.add_algorithm("LRU", LRUPageReplacement(3))
    
    for alg in ["FIFO", "LRU"]:
        simulator.reset()
        simulator.set_algorithm(alg)
        simulator.simulate_reference_string(test_addresses)
        stats = simulator.get_statistics()
        print(f"{alg}: {stats['page_faults']} page faults")
