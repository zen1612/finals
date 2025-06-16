import random
from collections import OrderedDict
from typing import List, Tuple, Dict

class Frame:
    """Represents a physical memory frame"""
    def __init__(self, page: int = -1, last_used: int = 0):
        self.page = page
        self.last_used = last_used
        self.valid = page != -1

class TLBEntry:
    """Translation Lookaside Buffer entry"""
    def __init__(self, virtual_page: int, physical_frame: int, timestamp: int = 0):
        self.virtual_page = virtual_page
        self.physical_frame = physical_frame
        self.timestamp = timestamp

class VirtualMemorySimulator:
    """Virtual Memory Simulator with address translation and TLB"""
    
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
        """Handle page fault using custom algorithm and return physical frame number"""
        if self.frame_count < self.physical_frames:
            # Free frame available
            frame_index = self.frame_count
            self.frames[frame_index] = Frame(virtual_page, self.current_time)
            self.frame_count += 1
        else:
            # Need to evict a page using custom algorithm
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
    
    def get_stats(self) -> Dict:
        """Return current statistics"""
        return {
            'page_faults': self.page_faults,
            'tlb_hits': self.tlb_hits,
            'tlb_misses': self.tlb_misses,
            'memory_accesses': self.memory_accesses,
            'page_fault_rate': self.page_faults / self.memory_accesses if self.memory_accesses > 0 else 0,
            'tlb_hit_rate': self.tlb_hits / self.memory_accesses if self.memory_accesses > 0 else 0
        }

class PageReplacementSimulator:
    """Simulator for different page replacement algorithms"""
    
    @staticmethod
    def fifo(pages: List[int], capacity: int) -> int:
        """FIFO page replacement algorithm"""
        frames = []
        page_faults = 0
        
        for page in pages:
            if page not in frames:
                page_faults += 1
                if len(frames) < capacity:
                    frames.append(page)
                else:
                    frames.pop(0)  # Remove first (oldest)
                    frames.append(page)
        
        return page_faults
    
    @staticmethod
    def lru(pages: List[int], capacity: int) -> int:
        """LRU page replacement algorithm"""
        frames = OrderedDict()
        page_faults = 0
        
        for page in pages:
            if page in frames:
                # Move to end (most recently used)
                frames.move_to_end(page)
            else:
                page_faults += 1
                if len(frames) < capacity:
                    frames[page] = True
                else:
                    # Remove least recently used (first item)
                    frames.popitem(last=False)
                    frames[page] = True
        
        return page_faults
    
    @staticmethod
    def custom_algorithm(pages: List[int], capacity: int) -> int:
        """Custom algorithm: LRU among first 2 frames (based on your C code)"""
        frames = []
        last_used = {}
        page_faults = 0
        current_time = 0
        
        for page in pages:
            current_time += 1
            
            if page in frames:
                # Update last used time
                last_used[page] = current_time
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
        
        return page_faults

def main():
    """Main simulation function"""
    print("Virtual Memory Simulator")
    print("=" * 50)
    
    # Test reference strings
    test_cases = [
        {
            'name': 'Test Case 1',
            'reference_string': [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5],
            'frames': 3,
            'tlb_size': 4
        },
        {
            'name': 'Test Case 2', 
            'reference_string': [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2],
            'frames': 3,
            'tlb_size': 2
        },
        {
            'name': 'Test Case 3',
            'reference_string': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8],
            'frames': 4,
            'tlb_size': 6
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"Reference String: {test_case['reference_string']}")
        print(f"Physical Frames: {test_case['frames']}")
        print(f"TLB Size: {test_case['tlb_size']}")
        print("-" * 50)
        
        # Page Replacement Algorithms
        reference_string = test_case['reference_string']
        frames = test_case['frames']
        
        fifo_faults = PageReplacementSimulator.fifo(reference_string, frames)
        lru_faults = PageReplacementSimulator.lru(reference_string, frames)
        custom_faults = PageReplacementSimulator.custom_algorithm(reference_string, frames)
        
        print("Page Replacement Results:")
        print(f"  FIFO Page Faults: {fifo_faults}")
        print(f"  LRU Page Faults: {lru_faults}")
        print(f"  Custom Page Faults: {custom_faults}")
        
        # Address Translation with TLB
        vm = VirtualMemorySimulator(
            physical_frames=frames,
            tlb_size=test_case['tlb_size'],
            page_size=1024
        )
        
        # Convert page numbers to virtual addresses
        virtual_addresses = [page * 1024 for page in reference_string]
        
        # Process all addresses
        for vaddr in virtual_addresses:
            vm.virtual_to_physical_address(vaddr)
        
        # Get and display statistics
        stats = vm.get_stats()
        print("\nAddress Translation with TLB Results:")
        print(f"  Page Faults: {stats['page_faults']}")
        print(f"  TLB Hits: {stats['tlb_hits']}")
        print(f"  TLB Misses: {stats['tlb_misses']}")
        print(f"  Page Fault Rate: {stats['page_fault_rate']:.2%}")
        print(f"  TLB Hit Rate: {stats['tlb_hit_rate']:.2%}")

if __name__ == "__main__":
    main()
