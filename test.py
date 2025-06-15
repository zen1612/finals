import random
import time
from collections import OrderedDict
from typing import List, Tuple, Optional, Dict

class Frame:
    """Represents a physical memory frame"""
    def __init__(self, page: int = -1, last_used: int = 0):
        self.page = page
        self.last_used = last_used
        self.valid = page != -1
    
    def __repr__(self):
        return f"Frame(page={self.page}, last_used={self.last_used})"

class TLBEntry:
    """Translation Lookaside Buffer entry"""
    def __init__(self, virtual_page: int, physical_frame: int, timestamp: int = 0):
        self.virtual_page = virtual_page
        self.physical_frame = physical_frame
        self.timestamp = timestamp
    
    def __repr__(self):
        return f"TLB({self.virtual_page}->{self.physical_frame})"

class VirtualMemorySimulator:
    """Complete Virtual Memory Simulator with address translation and TLB"""
    
    def __init__(self, physical_frames: int = 3, tlb_size: int = 4, page_size: int = 4096):
        self.physical_frames = physical_frames
        self.tlb_size = tlb_size
        self.page_size = page_size
        
        # Physical memory (frames)
        self.frames = [Frame() for _ in range(physical_frames)]
        self.frame_count = 0
        
        # Translation Lookaside Buffer
        self.tlb = []
        
        # Page table (virtual page -> physical frame mapping)
        self.page_table = {}
        
        # Statistics
        self.page_faults = 0
        self.tlb_hits = 0
        self.tlb_misses = 0
        self.memory_accesses = 0
        self.current_time = 0
        
        # Logs for analysis
        self.access_log = []
    
    def virtual_to_physical_address(self, virtual_addr: int) -> Tuple[int, bool, bool]:
        """
        Translate virtual address to physical address
        Returns: (physical_address, tlb_hit, page_fault)
        """
        self.memory_accesses += 1
        self.current_time += 1
        
        # Extract page number and offset
        virtual_page = virtual_addr // self.page_size
        offset = virtual_addr % self.page_size
        
        # Check TLB first
        tlb_hit = False
        physical_frame = None
        
        for entry in self.tlb:
            if entry.virtual_page == virtual_page:
                physical_frame = entry.physical_frame
                entry.timestamp = self.current_time
                tlb_hit = True
                self.tlb_hits += 1
                break
        
        if not tlb_hit:
            self.tlb_misses += 1
            
            # Check page table
            if virtual_page in self.page_table:
                physical_frame = self.page_table[virtual_page]
                page_fault = False
            else:
                # Page fault - need to load page
                physical_frame = self._handle_page_fault(virtual_page)
                page_fault = True
                self.page_faults += 1
            
            # Update TLB
            self._update_tlb(virtual_page, physical_frame)
        else:
            page_fault = False
        
        # Calculate physical address
        physical_addr = physical_frame * self.page_size + offset
        
        # Log the access
        self.access_log.append({
            'virtual_addr': virtual_addr,
            'virtual_page': virtual_page,
            'physical_addr': physical_addr,
            'physical_frame': physical_frame,
            'tlb_hit': tlb_hit,
            'page_fault': page_fault,
            'time': self.current_time
        })
        
        return physical_addr, tlb_hit, page_fault
    
    def _update_tlb(self, virtual_page: int, physical_frame: int):
        """Update TLB with new translation"""
        # Remove existing entry if present
        self.tlb = [entry for entry in self.tlb if entry.virtual_page != virtual_page]
        
        # Add new entry
        new_entry = TLBEntry(virtual_page, physical_frame, self.current_time)
        self.tlb.append(new_entry)
        
        # Maintain TLB size (LRU eviction)
        if len(self.tlb) > self.tlb_size:
            self.tlb.sort(key=lambda x: x.timestamp)
            self.tlb = self.tlb[1:]  # Remove oldest
    
    def _handle_page_fault(self, virtual_page: int) -> int:
        """Handle page fault and return physical frame number"""
        if self.frame_count < self.physical_frames:
            # Free frame available
            frame_index = self.frame_count
            self.frames[frame_index] = Frame(virtual_page, self.current_time)
            self.frame_count += 1
        else:
            # Need to evict a page
            frame_index = self._find_eviction_candidate()
            old_page = self.frames[frame_index].page
            
            # Remove old mapping
            if old_page in self.page_table:
                del self.page_table[old_page]
            
            # Remove from TLB
            self.tlb = [entry for entry in self.tlb if entry.virtual_page != old_page]
            
            # Load new page
            self.frames[frame_index] = Frame(virtual_page, self.current_time)
        
        # Update page table
        self.page_table[virtual_page] = frame_index
        return frame_index
    
    def _find_eviction_candidate(self) -> int:
        """Custom algorithm: LRU among first 2 frames (FIFO + LRU hybrid)"""
        min_pages = min(2, self.frame_count)
        oldest_time = float('inf')
        candidate_index = 0
        
        for i in range(min_pages):
            if self.frames[i].last_used < oldest_time:
                oldest_time = self.frames[i].last_used
                candidate_index = i
        
        return candidate_index

class PageReplacementSimulator:
    """Simulator for different page replacement algorithms"""
    
    @staticmethod
    def fifo(pages: List[int], capacity: int) -> Tuple[int, List[str]]:
        """FIFO page replacement algorithm"""
        frames = []
        page_faults = 0
        log = []
        
        for page in pages:
            if page not in frames:
                page_faults += 1
                if len(frames) < capacity:
                    frames.append(page)
                else:
                    frames.pop(0)  # Remove first (oldest)
                    frames.append(page)
                log.append(f"Page {page}: FAULT -> Frames: {frames.copy()}")
            else:
                log.append(f"Page {page}: HIT -> Frames: {frames.copy()}")
        
        return page_faults, log
    
    @staticmethod
    def lru(pages: List[int], capacity: int) -> Tuple[int, List[str]]:
        """LRU page replacement algorithm"""
        frames = OrderedDict()
        page_faults = 0
        log = []
        
        for page in pages:
            if page in frames:
                # Move to end (most recently used)
                frames.move_to_end(page)
                log.append(f"Page {page}: HIT -> Frames: {list(frames.keys())}")
            else:
                page_faults += 1
                if len(frames) < capacity:
                    frames[page] = True
                else:
                    # Remove least recently used (first item)
                    frames.popitem(last=False)
                    frames[page] = True
                log.append(f"Page {page}: FAULT -> Frames: {list(frames.keys())}")
        
        return page_faults, log
    
    @staticmethod
    def custom_algorithm(pages: List[int], capacity: int) -> Tuple[int, List[str]]:
        """Custom algorithm: LRU among first 2 frames (based on your C code)"""
        frames = []
        last_used = {}
        page_faults = 0
        log = []
        current_time = 0
        
        for page in pages:
            current_time += 1
            
            if page in frames:
                # Update last used time
                last_used[page] = current_time
                log.append(f"Page {page}: HIT -> Frames: {frames.copy()}")
            else:
                # Page fault
                page_faults += 1
                
                if len(frames) < capacity:
                    frames.append(page)
                    last_used[page] = current_time
                else:
                    # Find eviction candidate (LRU among first 2)
                    min_pages = min(2, len(frames))
                    oldest_time = float('inf')
                    evict_index = 0
                    
                    for i in range(min_pages):
                        frame_page = frames[i]
                        if last_used[frame_page] < oldest_time:
                            oldest_time = last_used[frame_page]
                            evict_index = i
                    
                    # Remove evicted page
                    evicted_page = frames.pop(evict_index)
                    del last_used[evicted_page]
                    
                    # Add new page
                    frames.append(page)
                    last_used[page] = current_time
                
                log.append(f"Page {page}: FAULT -> Frames: {frames.copy()}")
        
        return page_faults, log

def run_address_translation_demo():
    """Demonstrate address translation with TLB"""
    print("=== Virtual Memory Address Translation Demo ===")
    
    vm = VirtualMemorySimulator(physical_frames=3, tlb_size=4, page_size=1024)
    
    # Generate some virtual addresses
    virtual_addresses = [0, 1024, 2048, 512, 1024, 3072, 0, 1536, 2048]
    
    print(f"Page Size: {vm.page_size} bytes")
    print(f"Physical Frames: {vm.physical_frames}")
    print(f"TLB Size: {vm.tlb_size}")
    print("\nAddress Translation Results:")
    print("-" * 80)
    
    for vaddr in virtual_addresses:
        paddr, tlb_hit, page_fault = vm.virtual_to_physical_address(vaddr)
        vpage = vaddr // vm.page_size
        pframe = paddr // vm.page_size
        offset = vaddr % vm.page_size
        
        status = []
        if tlb_hit:
            status.append("TLB HIT")
        else:
            status.append("TLB MISS")
        if page_fault:
            status.append("PAGE FAULT")
        
        print(f"Virtual: {vaddr:5d} (page {vpage}, offset {offset:3d}) -> "
              f"Physical: {paddr:5d} (frame {pframe}) [{', '.join(status)}]")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total Memory Accesses: {vm.memory_accesses}")
    print(f"Page Faults: {vm.page_faults}")
    print(f"TLB Hits: {vm.tlb_hits}")
    print(f"TLB Misses: {vm.tlb_misses}")
    print(f"Page Fault Rate: {vm.page_faults/vm.memory_accesses:.2%}")
    print(f"TLB Hit Rate: {vm.tlb_hits/vm.memory_accesses:.2%}")
    
    # Show current TLB state
    print(f"\nCurrent TLB State:")
    for i, entry in enumerate(vm.tlb):
        print(f"  TLB[{i}]: Virtual Page {entry.virtual_page} -> Physical Frame {entry.physical_frame}")

def run_page_replacement_demo():
    """Demonstrate page replacement algorithms"""
    print("\n\n=== Page Replacement Algorithms Demo ===")
    
    # Test reference string
    reference_string = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    capacity = 3
    
    print(f"Reference String: {reference_string}")
    print(f"Number of Frames: {capacity}")
    print("-" * 60)
    
    # Test FIFO
    print("\n1. FIFO Algorithm:")
    fifo_faults, fifo_log = PageReplacementSimulator.fifo(reference_string, capacity)
    for entry in fifo_log:
        print(f"   {entry}")
    print(f"   Total Page Faults: {fifo_faults}")
    
    # Test LRU
    print("\n2. LRU Algorithm:")
    lru_faults, lru_log = PageReplacementSimulator.lru(reference_string, capacity)
    for entry in lru_log:
        print(f"   {entry}")
    print(f"   Total Page Faults: {lru_faults}")
    
    # Test Custom Algorithm
    print("\n3. Custom Algorithm (LRU among first 2 frames):")
    custom_faults, custom_log = PageReplacementSimulator.custom_algorithm(reference_string, capacity)
    for entry in custom_log:
        print(f"   {entry}")
    print(f"   Total Page Faults: {custom_faults}")
    
    # Summary
    print(f"\n=== Algorithm Comparison ===")
    print(f"FIFO Page Faults: {fifo_faults}")
    print(f"LRU Page Faults: {lru_faults}")
    print(f"Custom Page Faults: {custom_faults}")

def run_comprehensive_simulation():
    """Run a comprehensive simulation with various scenarios"""
    print("\n\n=== Comprehensive Virtual Memory Simulation ===")
    
    # Test different scenarios
    scenarios = [
        {"name": "Sequential Access", "pattern": list(range(0, 20)), "description": "Sequential page access"},
        {"name": "Random Access", "pattern": [random.randint(0, 15) for _ in range(20)], "description": "Random page access"},
        {"name": "Locality Pattern", "pattern": [1,1,2,2,1,3,3,1,4,4,1,2,5,5,2,1], "description": "Good temporal locality"},
    ]
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Description: {scenario['description']}")
        print(f"Access Pattern: {scenario['pattern']}")
        
        # Convert pages to virtual addresses (assuming page size 1024)
        virtual_addrs = [page * 1024 for page in scenario['pattern']]
        
        # Create fresh simulator
        vm = VirtualMemorySimulator(physical_frames=4, tlb_size=6, page_size=1024)
        
        # Run simulation
        for vaddr in virtual_addrs:
            vm.virtual_to_physical_address(vaddr)
        
        # Print results
        print(f"Results:")
        print(f"  Memory Accesses: {vm.memory_accesses}")
        print(f"  Page Faults: {vm.page_faults}")
        print(f"  TLB Hits: {vm.tlb_hits}")
        print(f"  Page Fault Rate: {vm.page_faults/vm.memory_accesses:.2%}")
        print(f"  TLB Hit Rate: {vm.tlb_hits/vm.memory_accesses:.2%}")

def interactive_simulator():
    """Interactive simulator for user input"""
    print("\n\n=== Interactive Virtual Memory Simulator ===")
    print("Enter virtual addresses (space-separated) or 'quit' to exit:")
    
    vm = VirtualMemorySimulator(physical_frames=3, tlb_size=4, page_size=1024)
    
    while True:
        try:
            user_input = input("\nVirtual addresses: ").strip()
            if user_input.lower() == 'quit':
                break
            
            addresses = list(map(int, user_input.split()))
            
            print("\nTranslation Results:")
            print("-" * 50)
            
            for vaddr in addresses:
                paddr, tlb_hit, page_fault = vm.virtual_to_physical_address(vaddr)
                vpage = vaddr // vm.page_size
                
                status = "TLB HIT" if tlb_hit else "TLB MISS"
                if page_fault:
                    status += ", PAGE FAULT"
                
                print(f"{vaddr:5d} -> {paddr:5d} (page {vpage}) [{status}]")
            
            print(f"\nCurrent Stats - Faults: {vm.page_faults}, "
                  f"TLB Hit Rate: {vm.tlb_hits/vm.memory_accesses:.1%}")
                  
        except ValueError:
            print("Invalid input. Please enter space-separated integers.")
        except KeyboardInterrupt:
            break
    
    print("\nFinal Statistics:")
    print(f"Total Accesses: {vm.memory_accesses}")
    print(f"Page Faults: {vm.page_faults}")
    print(f"TLB Hit Rate: {vm.tlb_hits/vm.memory_accesses:.2%}")

if __name__ == "__main__":
    # Run all demonstrations
    run_address_translation_demo()
    run_page_replacement_demo()
    run_comprehensive_simulation()
    
    # Uncomment the line below for interactive mode
    # interactive_simulator()
