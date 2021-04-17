// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GPU_TLB_H
#define GPU_TLB_H

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <queue>
#include <set>

#include "mshr.h"

#define GLOBAL_SPACE 0
#define TEXTURE_SPACE 1
#define SHARED_SPACE 2
#define OTHER_SPACE 3

class  PageWalker;
class  PageWalkSubsystem;

class mem_fetch;

class mmu;

class shader_core_stats;

class memory_stats_t;

class memory_partition_unit;

struct memory_config;

class dram_cmd;

class page_table;

class tlb_fetch;

enum tlb_request_status {
  TLB_HIT,
  TLB_HIT_RESERVED,
  TLB_PENDING,
  TLB_MISS,
  TLB_MSHR_FAIL,
  TLB_BACKPRESSURE_MISS,
  TLB_FLUSH,
  TLB_FAULT,
  TLB_PAGE_FAULT,
  TLB_NOFAULT,
  TLB_VIOLATION,
  TLB_MEMORY_RESERVATION_FAIL,
  COALESCED_TLB
};

class tlb_tag_array 
{
  private:
    std::list<new_addr_type>* tag_array;
    std::list<new_addr_type>** l2_tag_array;
    std::list<new_addr_type>** per_app_l2_tag_array[4];
    mshr_table m_mshrs;

  public:
    // std::queue<mem_fetch*> page_walk_request_queue;
    // std::vector<std::queue<mem_fetch*>> per_walker_page_walk_queue;
    unsigned in_flight_page_walks;

    PageWalkSubsystem* page_walk_subsystem;
    PageWalkSubsystem* per_app_page_walk_subsystem[4];

    bool stall;
    bool stall_app[4];

    memory_partition_unit** m_memory_partition;
    page_table* root;
    mmu* m_page_manager;
    shader_core_stats* m_stat;
    tlb_tag_array** l1_tlb;
    const memory_config* m_config;
    tlb_tag_array* m_shared_tlb;
    unsigned tlb_level;
    memory_stats_t* m_mem_stats;
    std::list<tlb_fetch*> request_queue;
    std::list<tlb_fetch*> per_app_failed_request_queue[4];
    int m_core_id;

    unsigned m_access, m_miss, m_ways, m_entries;
    bool isL2TLB;

    /* Unnecessary but unavoidable */
    std::map<int,std::set<new_addr_type>*> promoted_pages;

    /* Constructors and Destructor */
    tlb_tag_array(const memory_config* config, shader_core_stats* stat,
        mmu* page_manager, tlb_tag_array* shared_tlb, int core_id);
    tlb_tag_array(const memory_config* config, shader_core_stats* stat,
        mmu* page_manager, bool isL2TLB, memory_stats_t* mem_stat,
        memory_partition_unit** mem_partition);
    ~tlb_tag_array();

    /* Methods */
    enum tlb_request_status probe(new_addr_type addr,
        unsigned accessor, mem_fetch * mf);
    bool access(tlb_fetch* tf);
    bool reaccess(tlb_fetch* tf);

    new_addr_type get_tlbreq_addr(mem_fetch * mf);
    void set_l1_tlb(int coreID, tlb_tag_array* l1);
    int pump_failed_request_queue();

    void fill(new_addr_type addr, mem_fetch* mf);
    void fill(new_addr_type addr, unsigned accessor, mem_fetch* mf);

    bool request_shared_tlb(new_addr_type addr,
        unsigned accessor, mem_fetch* mf);
    /* L2 TLB specific methods */
    void cycle(); 
    void fill_into_l1_tlb(new_addr_type addr, mem_fetch* mf);
    void l2_fill(new_addr_type addr, unsigned accessor, mem_fetch* mf);

    /* Unnecessary but unavoidable */
    int promotion(new_addr_type va, int appID);
    // unsigned flush(int appID)
    // {
    //   return 0;
    // }
    // unsigned flush();
    unsigned flush(int appID);
    unsigned flush();
    bool demote_page(new_addr_type va, int appID)
    {
      return false;
    }

    /* Accessor Methods */
    tlb_tag_array* get_shared_tlb() {
      return m_shared_tlb;
    }
    memory_stats_t* get_mem_stat() {
      return m_mem_stats;
    }

    /* Page Walk Cache. Will be removed once the page walk class comes up */
    void pw_cache_print();
    void tlb_print();

    /* Static Queue for fixed latency page walks */
    /* Must be moved to page walker class or page walk queue class later */
    std::list<mem_fetch*>* static_queue;
    std::list<unsigned long long>* static_queue_time;
    void put_mf_to_static_queue(mem_fetch * mf);
    void clear_static_queue();

    /* Pratheek: proper mshr support */
    bool access_ready() const {
      return m_mshrs.access_ready();
    }
    mem_fetch *next_access() {
      return m_mshrs.next_access();
    }

    /* For tracking page walks at L2 TLB */
    std::set<new_addr_type> page_walk_tracker;
    /* For tracking stalled warps at L1 TLB */
    std::map<new_addr_type, unsigned> stalled_warps;
    
    unsigned tlb_state_tracker[4];

    /* MASK Implementation */
    std::list<new_addr_type> mask_bypass_tlb;
    void mask_token_reassignment();
    void mask_tlb_fill(new_addr_type addr, mem_fetch* mf);
    void fill_bypass_cache(new_addr_type addr, unsigned accessor, mem_fetch* mf);
    new_addr_type get_key(new_addr_type addr, unsigned appid);
    unsigned get_app_from_key(new_addr_type key);
    /* bool mask_tlb_fill_bypass_check(mem_fetch* mf); */

};

class tlb_fetch
{
  public:
    tlb_fetch(tlb_tag_array* origin_tlb, mem_fetch* mf, new_addr_type addr, 
        unsigned accessor, unsigned ready_cycle);
    mem_fetch* get_mf();

    unsigned ready_cycle;

    /* private: */
    mem_fetch* mf;
    tlb_tag_array* origin_tlb;
    new_addr_type addr;
    unsigned accessor;
};

#endif
