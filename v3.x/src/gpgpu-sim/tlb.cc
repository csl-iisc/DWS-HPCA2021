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

#include "stat-tool.h"
#include <assert.h>
#include "tlb.h"
#include "dram.h"
#include "gpu-sim.h"
#include <stdlib.h>
#include "mem_latency_stat.h"
#include "l2cache.h"
#include <map>
#include "pagewalk.h"

#define COALESCE_DEBUG 0
#define COALESCE_DEBUG_SHORT 0
#define PWCACHE_DEBUG 0
#define MERGE_DEBUG 0

extern int gpu_sms;

tlb_tag_array::tlb_tag_array(const memory_config *config,
                             shader_core_stats *stat, mmu *page_manager,
                             tlb_tag_array *shared_tlb, int core_id) : m_mshrs(12, 4)
{

  m_config = config;
  m_stat = stat;
  m_page_manager = page_manager;
  m_shared_tlb = shared_tlb;
  m_core_id = core_id;

  m_mem_stats = shared_tlb->get_mem_stat();

  tlb_level = 0;
  isL2TLB = false;

  tag_array = new std::list<new_addr_type>;
  root = page_manager->get_page_table_root();

  if (shared_tlb != NULL)
    printf("Assigned Shared TLB at %x\n", m_shared_tlb);

  m_access = 0;
  m_miss = 0;
}

tlb_tag_array::tlb_tag_array(const memory_config *config,
                             shader_core_stats *stat, mmu *page_manager, bool isL2TLB,
                             memory_stats_t *mem_stat, memory_partition_unit **memory_partition) : m_mshrs(192, 24)
{

  m_config = config;
  m_stat = stat;
  m_page_manager = page_manager;
  m_mem_stats = mem_stat;
  m_memory_partition = memory_partition;

  m_shared_tlb = NULL;
  m_ways = m_config->l2_tlb_ways;
  m_entries = m_config->l2_tlb_ways == 0 ? 0 : m_config->l2_tlb_entries / m_config->l2_tlb_ways;
  if (m_config->vm_config == VM_PER_APP_TLB ||
      m_config->vm_config == VM_DWS_PER_APP_TLB ||
      m_config->vm_config == VM_PER_APP_VM_SUBSYSTEM)
  {
    for (int n = 0; n < ConfigOptions::n_apps; n++)
    {
      unsigned appid = App::get_app_id(n);
      per_app_l2_tag_array[appid] =
          new std::list<new_addr_type> *[m_entries];
      for (int i = 0; i < m_entries; i++)
      {
        per_app_l2_tag_array[appid][i] = new std::list<new_addr_type>;
      }
    }
  }
  else
  {
    l2_tag_array = new std::list<new_addr_type> *[m_entries];
    for (int i = 0; i < m_entries; i++)
    {
      l2_tag_array[i] = new std::list<new_addr_type>;
    }
  }
  if (m_config->vm_config == VM_PER_APP_WALK_SUBSYSTEM ||
      m_config->vm_config == VM_PER_APP_VM_SUBSYSTEM)
  {
    for (int n = 0; n < ConfigOptions::n_apps; n++)
    {
      unsigned appid = App::get_app_id(n);
      per_app_page_walk_subsystem[appid] =
          new PageWalkSubsystem(this, page_manager, config, mem_stat);
    }
  }
  else
  {
    page_walk_subsystem =
        new PageWalkSubsystem(this, page_manager, config, mem_stat);
  }
  stall = false;
  for (int n = 0; n < ConfigOptions::n_apps; n++)
  {
    unsigned appid = App::get_app_id(n);
    stall_app[appid] = false;
    tlb_state_tracker[appid] = 0;
  }

  l1_tlb = new tlb_tag_array *[gpu_sms];
  m_page_manager->set_L2_tlb(this);

  if (m_config->tlb_fixed_latency_enabled ||
      m_config->second_app_ideal_page_walk)
  {
    static_queue = new std::list<mem_fetch *>();
    static_queue_time = new std::list<unsigned long long>();
  }

  tlb_level = 0;
  isL2TLB = true;

  in_flight_page_walks = 0;
  m_access = 0;
  m_miss = 0;

  /* MASK */
  for (int n = 0; n < ConfigOptions::n_apps; n++)
  {
    unsigned appID = App::get_app_id(n);
    App *app = App::get_app(appID);
    for (int i = 0; i < 4000; i++)
    {
      app->wid_tokens[i] = true;
      app->wid_epoch_accesses[i] = 0;
      app->wid_epoch_hit[i] = 0;
      app->wid_epoch_miss[i] = 0;
      app->epoch_accesses = 0;
      app->epoch_miss = 0;
      app->epoch_hit = 0;
    }
    /* app->tokens = (float) m_config->mask_initial_tokens; */
    app->tokens = 1.0;
  }
}

/* For static TLB latency */
void tlb_tag_array::put_mf_to_static_queue(mem_fetch *mf)
{
  if (static_queue == NULL)
  {
    printf("ERROR: Static TLB latency queue is not initiated\n");
  }
  else
  {
    static_queue->push_back(mf);
    static_queue_time->push_back(gpu_sim_cycle + gpu_tot_sim_cycle);
  }
}

/* For static latency, remove anything that are done.
 * Called before TLB probe */
void tlb_tag_array::clear_static_queue()
{
  mem_fetch *top;
  while (!static_queue->empty())
  {
    if ((static_queue_time->front() + m_config->tlb_fixed_latency) <
        (gpu_sim_cycle + gpu_tot_sim_cycle))
    {
      mem_fetch *mf = static_queue->front();
      mf->done_tlb_req(mf);
      static_queue->pop_front();
      static_queue_time->pop_front();
    }
    else
    {
      break;
    }
  }
}

/* Rachata: Given an access, get the addess of the localtion of the tlb */
new_addr_type tlb_tag_array::get_tlbreq_addr(mem_fetch *mf)
{
  new_addr_type return_addr = root->parse_pa(mf);
  if (COALESCE_DEBUG)
    printf("Generating TLBreq for mf = %x, level = %d, return addr = %x\n",
           mf->get_addr(), mf->get_tlb_depth_count(), return_addr);
  return return_addr;
}

// Right now multi-page-size fill only support baseline and MASK
void tlb_tag_array::fill(new_addr_type addr, mem_fetch *mf)
{

  /* Does not have to fill if we always return TLB HIT(speeding things up)*/
  if (m_config->vm_config == VM_IDEAL_TLB)
    return;

  new_addr_type key;
  key = get_key(addr, mf->get_appID());

  /* Remove the current entry, put this in the front of the queue */
  tag_array->remove(key);

  if (m_mshrs.probe(key))
  {
    m_mshrs.mark_ready(key);
  }

  std::list<new_addr_type> *active_tag_array = tag_array;

  if (active_tag_array->size() >= m_config->tlb_size)
  {
    if (m_config->tlb_replacement_policy == 0)
    {
      active_tag_array->pop_back();
    }
    /* Other replacement policies go here */
    else
      active_tag_array->pop_back();
  }
  active_tag_array->push_front(key);
}

// New coalesce routine called from Allocator Hub. Note that coalescing is decided from the allocator hence it does not have to
// handle data movements from here.
int tlb_tag_array::promotion(new_addr_type va, int appID)
{
  if (MERGE_DEBUG)
    printf("Got a promotion call for appID = %d, va = %x, num_apps = %d\n", appID, va, ConfigOptions::n_apps);
  App *app = App::get_app(appID);

  new_addr_type base_addr = va / (*(m_config->page_sizes))[m_config->page_sizes->size() - 2];

  if (MERGE_DEBUG || COALESCE_DEBUG || COALESCE_DEBUG_SHORT)
    printf(
        "Putting the base page %x in the promoted_pages list. (Shifted base_VA with appID = %x).\n",
        base_addr,
        (base_addr * (*(m_config->page_sizes))[m_config->page_sizes->size() - 2]));
  m_mem_stats->coalesced_tried++;
  app->coalesced_tried_app++;

  promoted_pages[appID]->insert(
      (base_addr * (*(m_config->page_sizes))[m_config->page_sizes->size() - 2]));

  if (MERGE_DEBUG || COALESCE_DEBUG || COALESCE_DEBUG_SHORT)
  {
    printf("Current content of the promoted pages for app %d are: {", appID);
    for (std::set<new_addr_type>::iterator itr = promoted_pages[appID]->begin();
         itr != promoted_pages[appID]->end(); itr++)
      printf("%x, ", *itr);
    printf("}\n");
  }
}

/* Only called from L1 TLB to L2 TLB */
void tlb_tag_array::set_l1_tlb(int coreID, tlb_tag_array *l1)
{
  printf("Setting L1 TLB pointer for the shared TLB for SM %d\n", coreID);
  l1_tlb[coreID] = l1;
}

void tlb_tag_array::l2_fill(new_addr_type addr,
                            unsigned accessor, mem_fetch *mf)
{
  m_shared_tlb->fill(addr, accessor, mf);
}

/* For L2 TLB only */
void tlb_tag_array::fill(new_addr_type addr, unsigned accessor,
                         mem_fetch *mf)
{

  /* Does not have to fill if we always return TLB HIT(speeding things up) */
  if (m_config->vm_config == VM_IDEAL_TLB)
    return;

  new_addr_type key =
      get_key(addr, mf->get_appID());

  if (m_config->vm_config == VM_MASK || m_config->vm_config == VM_MASK_PLUS_DWS)
  {
    ;
  }
  else
  {
    m_mshrs.mark_ready(key);
    page_walk_tracker.erase(key);
  }

  unsigned index;
  index = (key) & (m_entries - 1);

  bool evicted = false;

  std::list<new_addr_type> **correct_tag_array;
  if (m_config->vm_config == VM_PER_APP_TLB ||
      m_config->vm_config == VM_DWS_PER_APP_TLB ||
      m_config->vm_config == VM_PER_APP_VM_SUBSYSTEM)
  {
    correct_tag_array = per_app_l2_tag_array[accessor];
  }
  else
  {
    correct_tag_array = l2_tag_array;
  }

  /* Remove the current entry, put this in the front of the queue */
  correct_tag_array[index]->remove(key);

  new_addr_type removed_entry;
  if (correct_tag_array[index]->size() >= (m_ways))
  {
    evicted = true;
    if (m_config->tlb_replacement_policy == 0)
    {
      removed_entry = correct_tag_array[index]->back();
      unsigned removed_entry_app = get_app_from_key(removed_entry); //removed_entry >> 36;
      tlb_state_tracker[removed_entry_app]--;
      correct_tag_array[index]->pop_back();
    }
    /* Other replacement policies go here */
    else
    {
      removed_entry = correct_tag_array[index]->back();
      correct_tag_array[index]->pop_back();
    }
  }

  correct_tag_array[index]->push_front(key);
  unsigned inserted_entry_app = get_app_from_key(key); //key >> 36;
  tlb_state_tracker[inserted_entry_app]++;

  m_mem_stats->tlb_fills[accessor]++;
}

/* call only from MASK code */
void tlb_tag_array::fill_bypass_cache(new_addr_type addr, unsigned accessor,
                                      mem_fetch *mf)
{

  new_addr_type key =
      get_key(addr, mf->get_appID());

  assert(m_config->vm_config == VM_MASK || m_config->vm_config == VM_MASK_PLUS_DWS);

  unsigned index;
  index = (key) & (m_entries - 1);

  bool evicted = false;

  /* Remove the current entry, put this in the front of the queue */
  mask_bypass_tlb.remove(key);

  new_addr_type removed_entry;
  /* 32 is the size of the bypass TLB */
  if (mask_bypass_tlb.size() >= (32))
  {
    evicted = true;
    if (m_config->tlb_replacement_policy == 0)
    {
      removed_entry = mask_bypass_tlb.back();
      // unsigned removed_entry_app = get_app_from_key(removed_entry); //removed_entry >> 36;
      /* tlb_state_tracker[removed_entry_app]--; */
      mask_bypass_tlb.pop_back();
    }
    /* Other replacement policies go here */
    else
    {
      removed_entry = mask_bypass_tlb.back();
      mask_bypass_tlb.pop_back();
    }
  }

  mask_bypass_tlb.push_front(key);
  // unsigned inserted_entry_app = key >> 36;
  /* tlb_state_tracker[inserted_entry_app]++; */

  m_mem_stats->tlb_fills[accessor]++;
}

enum tlb_request_status tlb_tag_array::probe(
    new_addr_type addr, unsigned accessor, mem_fetch *mf)
{

  if (m_config->vm_config == VM_IDEAL_TLB)
    return TLB_HIT;
  if (m_config->vm_config == VM_SECOND_APP_IDEAL_TLB)
  {
    /* Please note we have assumptions on the app IDs starting at 1 and then
     * being sequential from there.*/
    if (mf->get_appID() == 2)
      return TLB_HIT;
  }

  App *app = App::get_app(mf->get_appID());
  new_addr_type searched_key;
  searched_key = get_key(addr, mf->get_appID());

  /* L1 TLB */
  if (m_shared_tlb != NULL)
  {

    // If fixed TLB latency is used, clear out any finished requests
    if (m_config->tlb_fixed_latency_enabled)
    {
      m_shared_tlb->clear_static_queue();
    }

    new_addr_type key = searched_key;

    std::list<new_addr_type>::iterator findIter =
        std::find(tag_array->begin(), tag_array->end(), key);

    m_stat->tlb_access++;
    app->tlb_access_app++;

    if (findIter != tag_array->end())
    {
      tag_array->remove(key);
      tag_array->push_front(key); // insert at MRU
      m_stat->tlb_hit++;
      app->tlb_hit_app++;
      return TLB_HIT;
    }
    else
    {
      bool mshr_hit = m_mshrs.probe(key);
      bool mshr_avail = !m_mshrs.full(key);
      if (mshr_hit && mshr_avail)
      {
        m_mshrs.add(key, mf);
        // app->debug_tlb_mshrs++;
        m_stat->tlb_mshr_hit++;
        app->tlb_mshr_hit_app++;
        return TLB_MISS;
      }
      else if (mshr_hit && !mshr_avail)
      {
        m_stat->tlb_mshr_fail++;
        app->tlb_mshr_fail_app++;
        return TLB_MSHR_FAIL;
      }

      assert(!mshr_hit);
      if (!mshr_hit && mshr_avail)
      {
        if (request_shared_tlb(addr, accessor, mf))
        {
          m_mshrs.add(key, mf);
          m_stat->tlb_miss++;
          app->tlb_miss_app++;
          return TLB_MISS;
        }
        else
        {
          m_stat->tlb_bkpres_fail++;
          app->tlb_bkpres_fail_app++;
          return TLB_MSHR_FAIL;
        }
      }
      else
      {
        m_stat->tlb_mshr_fail++;
        app->tlb_mshr_fail_app++;
        return TLB_MSHR_FAIL;
      }
    }
  }
  /* L2 */
  else
  {

    new_addr_type key = searched_key; //small page
    new_addr_type index;

    index = key & (m_entries - 1);

    if (m_config->vm_config == VM_MASK || m_config->vm_config == VM_MASK_PLUS_DWS)
    {
      std::list<new_addr_type>::iterator findIter;
      findIter = std::find(mask_bypass_tlb.begin(),
                           mask_bypass_tlb.end(), key);
      if (findIter != mask_bypass_tlb.end()) //Found in the small page TLB
      {

        /* Fill into L1 TLB */
        tlb_tag_array *l1_tlb = mf->get_tlb();
        l1_tlb->fill(mf->get_addr(), mf);

        mask_bypass_tlb.remove(key);
        mask_bypass_tlb.push_front(key); // insert at MRU
        app->bypass_tlb_hit++;

        return TLB_HIT;
      }
      else
      {
        app->bypass_tlb_miss++;
      }
    }

    std::list<new_addr_type> **correct_tag_array;
    if (m_config->vm_config == VM_PER_APP_TLB ||
        m_config->vm_config == VM_DWS_PER_APP_TLB ||
        m_config->vm_config == VM_PER_APP_VM_SUBSYSTEM)
    {
      correct_tag_array = per_app_l2_tag_array[accessor];
    }
    else
    {
      correct_tag_array = l2_tag_array;
    }

    std::list<new_addr_type>::iterator findIter;
    findIter = std::find(correct_tag_array[index]->begin(),
                         correct_tag_array[index]->end(), key);

    if (findIter != correct_tag_array[index]->end()) //Found in the small page TLB
    {

      /* Fill into L1 TLB */
      tlb_tag_array *l1_tlb = mf->get_tlb();
      l1_tlb->fill(mf->get_addr(), mf);

      correct_tag_array[index]->remove(key);
      correct_tag_array[index]->push_front(key); // insert at MRU

      return TLB_HIT;
    }
    else //Not found
    {
      bool mshr_hit = m_mshrs.probe(key);
      bool mshr_avail = !m_mshrs.full(key);
      if (mshr_hit && mshr_avail)
      {
        m_mshrs.add(key, mf);
        return TLB_HIT_RESERVED;
      }
      if (!mshr_hit && mshr_avail)
      {
        if (m_config->vm_config == VM_PER_APP_WALK_SUBSYSTEM ||
            m_config->vm_config == VM_PER_APP_VM_SUBSYSTEM)
        {
          page_walk_subsystem = per_app_page_walk_subsystem[mf->get_appID()];
        }
        if (page_walk_subsystem->enqueue(mf))
        {
          new_addr_type key = get_key(addr, mf->get_appID());
          m_mshrs.add(key, mf);
          page_walk_tracker.insert(key);
          mf->get_tlb()->stalled_warps.insert(std::pair<new_addr_type, unsigned>(key, mf->get_wid()));
          return TLB_MISS;
        }
        else
        {
          return TLB_BACKPRESSURE_MISS;
        }
      }
      else
      {
        return TLB_MSHR_FAIL;
      }
    }
  }
}

bool tlb_tag_array::request_shared_tlb(new_addr_type addr,
                                       unsigned accessor, mem_fetch *mf)
{

  if (m_config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      m_config->vm_config == VM_MASK_PLUS_DWS ||
      m_config->vm_config == VM_DWS_PER_APP_TLB ||
      m_config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
  {
    if (m_shared_tlb->stall_app[mf->get_appID()])
    {
      m_mem_stats->l2_tlb_tot_backpressure_stalls++;
      m_mem_stats->l2_tlb_app_backpressure_stalls[mf->get_appID()]++;
      return false;
    }
  }
  else
  {
    if (m_shared_tlb->stall)
    {
      m_mem_stats->l2_tlb_tot_backpressure_stalls++;
      return false;
    }
  }

  unsigned ready_cycle =
      gpu_sim_cycle + gpu_tot_sim_cycle + m_config->l2_tlb_latency;
  tlb_fetch *tf = new tlb_fetch(this, mf, addr, accessor, ready_cycle);
  m_shared_tlb->request_queue.push_back(tf);

  return true;
}

/* L2 TLB only */
void tlb_tag_array::fill_into_l1_tlb(new_addr_type addr, mem_fetch *mf)
{

  tlb_tag_array *l1_t;
  new_addr_type key =
      get_key(addr, mf->get_appID());
  while (m_mshrs.access_ready())
  {
    mem_fetch *mshr_f = m_mshrs.next_access();
    l1_t = mshr_f->get_tlb();
    assert(l1_t->m_mshrs.probe(key));
    l1_t->fill(addr, mf);
  }
}

/* This method allows multiple apps to exist with QoS provisioned page walker */
/* partitioning */

int tlb_tag_array::pump_failed_request_queue()
{
  int appid = -1;
  unsigned long long ready_time = 100000000000;

  int ports = 0;
  while (ports < m_config->l2_tlb_ports)
  {
    appid = -1;
    for (int i = 0; i < 4; i++)
    {
      if (!per_app_failed_request_queue[i].empty())
      {
        tlb_fetch *tf = per_app_failed_request_queue[i].front();
        mem_fetch *mf = tf->mf;
        if (page_walk_subsystem->get_page_walk_queue_enqueue(mf) != -1)
        {
          if (ready_time > tf->ready_cycle)
          {
            appid = i;
            ready_time = tf->ready_cycle;
          }
        }
      }
    }

    if (appid != -1)
    {
      tlb_fetch *tf = per_app_failed_request_queue[appid].front();
      bool success = reaccess(tf);
      if (success)
      {
        per_app_failed_request_queue[appid].pop_front();
        /* Memory leak */
        ports++;
      }
      else
        break;
    }
    else
      break;
  }
  return ports;
}

/* Returns True if mem_fetch could be sent downstreams or added to MSHR*/
bool tlb_tag_array::reaccess(tlb_fetch *tf)
{
  mem_fetch *mf = tf->mf;
  new_addr_type key = get_key(mf->get_addr(), mf->get_appID());

  bool mshr_hit = m_mshrs.probe(key);
  bool mshr_avail = !m_mshrs.full(key);
  if (mshr_hit && mshr_avail)
  {
    m_mshrs.add(key, mf);
    return true;
  }

  if (!mshr_hit && mshr_avail)
  {
    if (page_walk_subsystem->enqueue(mf))
    {
      new_addr_type key = get_key(mf->get_addr(), mf->get_appID());
      m_mshrs.add(key, mf);
      page_walk_tracker.insert(key);
      mf->get_tlb()->stalled_warps.insert(std::pair<new_addr_type, unsigned>(key, mf->get_wid()));
      return true;
    }
    else
    {
      return false;
    }
  }
  else
  {
    return false;
  }
}

/* Returns True if access was a success and the next access in the
 * queue needs to be checked.*/
bool tlb_tag_array::access(tlb_fetch *tf)
{

  mem_fetch *mf = tf->mf;
  unsigned app_id = mf->get_appID();
  /* the probe function handles the downstream sending of page walk */
  tlb_request_status status = probe(mf->get_addr(), mf->get_appID(), mf);

  if (m_config->vm_config == VM_MASK || m_config->vm_config == VM_MASK_PLUS_DWS)
  {
    unsigned global_warp_id = mf->get_sid() * 100 + mf->get_wid();
    App::get_app(app_id)->epoch_accesses++;
    App::get_app(app_id)->wid_epoch_accesses[global_warp_id]++;
    if (status == TLB_HIT || status == TLB_HIT_RESERVED)
    {
      App::get_app(app_id)->epoch_hit++;
      App::get_app(app_id)->wid_epoch_hit[global_warp_id]++;
    }
    if (status == TLB_MISS)
    {
      App::get_app(app_id)->epoch_miss++;
      App::get_app(app_id)->wid_epoch_miss[global_warp_id]++;
    }
  }

  if (status == TLB_HIT)
  {
    m_mem_stats->l2_tlb_tot_hits++;
    m_mem_stats->l2_tlb_app_hits[app_id]++;
    return true;
  }

  else if (status == TLB_MISS)
  {
    m_mem_stats->l2_tlb_tot_misses++;
    m_mem_stats->l2_tlb_app_misses[app_id]++;
    return true;
  }
  else if (status == TLB_MSHR_FAIL)
  {
    m_mem_stats->l2_tlb_tot_mshr_fails++;
    m_mem_stats->l2_tlb_app_mshr_fails[app_id]++;
    /* DO NOT remove the page walk from the request queue
     * if the page walk is not enqueued */
    return false;
  }

  else if (status == TLB_HIT_RESERVED)
  {
    m_mem_stats->l2_tlb_tot_mshr_hits++;
    m_mem_stats->l2_tlb_app_mshr_hits[app_id]++;
    return true;
  }

  else if (status == TLB_BACKPRESSURE_MISS)
  {
    m_mem_stats->l2_tlb_tot_backpressure_fails++;
    m_mem_stats->l2_tlb_app_backpressure_fails[app_id]++;
    if (m_config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
        m_config->vm_config == VM_MASK_PLUS_DWS ||
        m_config->vm_config == VM_DWS_PER_APP_TLB ||
        m_config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
    {
      /* push into per app queue for stalled page walk requests */
      /* pop the request queue*/
      /* and continue (the for loop) */
      /* break for now */
      /* return false; */

      /* Return True because the access was a success because we have to
       * remove the request from the queue */
      per_app_failed_request_queue[app_id].push_back(tf);
      return true;
    }
    else
    {
      return false;
      ;
    }
  }

  else
  {
    assert(0 && "should not have reached here\n");
  }
}

/* This is to be called only on the shared TLB */
void tlb_tag_array::cycle()
{

  if (m_config->stealing_latency_enabled)
  {
    page_walk_subsystem->stealing_latency_resume();
  }

  if (m_config->vm_config == VM_PER_APP_WALK_SUBSYSTEM ||
      m_config->vm_config == VM_PER_APP_VM_SUBSYSTEM)
  {
    /* this stat segfaults because */
    /* TODO: take care of the stat for these configs */
    for(int i = 0; i < ConfigOptions::n_apps; i++) {
      unsigned appid = App::get_app_id(i);
      m_mem_stats->sum_num_page_walkers_assigned[appid] += per_app_page_walk_subsystem[appid]->current_allocation[appid];
    }
  }
  else
  {
    for (int n = 0; n < ConfigOptions::n_apps; n++)
    {
      unsigned appid = App::get_app_id(n);
      m_mem_stats->sum_num_page_walkers_assigned[appid] += page_walk_subsystem->current_allocation[appid];
      /* printf("%d %d\n", appid, page_walk_subsystem->current_allocation[appid]); */
    }
  }

  if (m_config->vm_config == VM_MASK || m_config->vm_config == VM_MASK_PLUS_DWS)
  {
    if (gpu_sim_cycle % m_config->mask_epoch_length == 0)
    {
      mask_token_reassignment();
    }
  }

  for (int n = 0; n < ConfigOptions::n_apps; n++)
  {
    unsigned appid = App::get_app_id(n);
    m_mem_stats->tlb_occupancy_sum[appid] += tlb_state_tracker[appid];
  }

  /* Take the latest ready tlb_fetch from the queue and probe it.
   * If probe hits, fill into L1 TLB
   * If probe misses, page walk on the mf
   * */

  unsigned ports = 0;

  /* TODO:REFACTORING
   * While we need support for mulit application QoS, we don't need it when
   * implementing BASELINE, so we ignore it for now. It is imperative to
   * get per app internal buffer for multi application support
   * */

  if (m_config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      m_config->vm_config == VM_MASK_PLUS_DWS ||
      m_config->vm_config == VM_DWS_PER_APP_TLB ||
      m_config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
  {

    int pumped = 0;
    ports = pump_failed_request_queue();
  }

  while (!request_queue.empty() && ports < m_config->l2_tlb_ports)
  {

    tlb_fetch *tf = request_queue.front();
    if (tf->ready_cycle > gpu_sim_cycle + gpu_tot_sim_cycle)
    {
      break;
    }

    m_mem_stats->l2_tlb_tot_accesses++;
    // m_mem_stats->l2_tlb_accesses[tf->mf->get_core()]++;

    mem_fetch *mf = tf->mf;
    unsigned app_id = mf->get_appID();
    m_mem_stats->l2_tlb_app_accesses[app_id]++;

    bool success = access(tf);

    if (success)
    {
      request_queue.pop_front();
      /* memory leak */
      /* Delete the tlb_fetch only if it has not been added to the per app
       * queue. This would require access() returning a tri-valued object */
      ports++;
    }
    else
    {
      break;
    }
  }

  if (m_config->second_app_ideal_page_walk)
  {
    clear_static_queue();
  }

  if (m_config->pw_cache_enable)
  {
    if (m_config->vm_config == VM_PER_APP_WALK_SUBSYSTEM ||
        m_config->vm_config == VM_PER_APP_VM_SUBSYSTEM)
    {
      for (int n = 0; n < ConfigOptions::n_apps; n++)
      {
        unsigned appid = App::get_app_id(n);
        per_app_page_walk_subsystem[appid]->service_page_walk_cache_queue();
      }
    }
    else
    {
      page_walk_subsystem->service_page_walk_cache_queue();
    }
  }
  return;
}

void tlb_tag_array::tlb_print()
{
  /* L1 TLB has pointer to shared TLB */
  if (m_shared_tlb != NULL)
  {
    std::list<new_addr_type>::iterator iter;
    for (iter = tag_array->begin(); iter != tag_array->end(); iter++)
    {
      printf("%p\t", *iter);
    }
    printf("\n");
    fflush(stdout);
  }
  /* L2 TLB */
  else
  {
    for (unsigned index = 0; index < m_entries; index++)
    {
      printf("index = %d\n", index);
      std::list<new_addr_type>::iterator iter;
      for (iter = l2_tag_array[index]->begin();
           iter != l2_tag_array[index]->end(); iter++)
      {
        printf("%p\t", *iter);
      }
      printf("\n");
      fflush(stdout);
    }
  }
}

tlb_fetch::tlb_fetch(tlb_tag_array *origin_tlb, mem_fetch *mf,
                     new_addr_type addr, unsigned accessor, unsigned ready_cycle)
{
  this->origin_tlb = origin_tlb;
  this->mf = mf;
  this->addr = addr;
  this->accessor = accessor;
  this->ready_cycle = ready_cycle;
}

mem_fetch *tlb_fetch::get_mf()
{
  return mf;
}

unsigned tlb_tag_array::flush()
{
  assert(m_shared_tlb && l2_tag_array);
  tag_array->clear();
}

unsigned tlb_tag_array::get_app_from_key(new_addr_type key)
{
  uint64_t appid = key & ((uint64_t) 3 << 62);
  appid = appid >> 62;
  return appid;
} 

new_addr_type tlb_tag_array::get_key(new_addr_type addr, unsigned appid)
{
  new_addr_type key = addr / (*(m_config->page_sizes))[m_config->page_sizes->size() - 1];
  assert((key & ((uint64_t) 3 << 62)) == 0);
  key = ((uint64_t) appid << 62) | key;
  return key;
}

// unsigned tlb_tag_array::flush(int appid)
// {
//   assert(appid == 1 || appid == 2);
//   assert(!m_shared_tlb);
//   assert(page_walk_subsystem);
//   page_walk_subsystem->flush(appid);
//   // unsigned appid = (unsigned) appid;
//   // assert(appid);
//   for (unsigned index = 0; index < m_entries; index++)
//   {
//     std::list<new_addr_type>::iterator iter;
//     for (iter = l2_tag_array[index]->begin(); iter != l2_tag_array[index]->end();)
//     {
//       new_addr_type key = *iter;
//       if (get_app_from_key(key) == appid)
//         iter = l2_tag_array[index]->erase(iter);
//       else
//         iter++;
//     }
//   }
//   tlb_state_tracker[appid] = 0;
// }

unsigned tlb_tag_array::flush(int appid)
{
  assert(appid == 1 || appid == 2);
  assert(!m_shared_tlb);
  // assert(page_walk_subsystem);
  if(m_config->vm_config == VM_PER_APP_VM_SUBSYSTEM || m_config->vm_config == VM_PER_APP_WALK_SUBSYSTEM)
  {
    per_app_page_walk_subsystem[appid]->flush(appid);
  }
  else
  {
    page_walk_subsystem->flush(appid);
  }
  
  std::list<new_addr_type> **correct_tag_array;
  if (m_config->vm_config == VM_PER_APP_TLB ||
      m_config->vm_config == VM_DWS_PER_APP_TLB ||
      m_config->vm_config == VM_PER_APP_VM_SUBSYSTEM)
  {
    correct_tag_array = per_app_l2_tag_array[appid];
  }
  else
  {
    correct_tag_array = l2_tag_array;
  }
  // unsigned appid = (unsigned) appid;
  // assert(appid);
  for (unsigned index = 0; index < m_entries; index++)
  {
    std::list<new_addr_type>::iterator iter;
    for (iter = correct_tag_array[index]->begin(); iter != correct_tag_array[index]->end();)
    {
      new_addr_type key = *iter;
      if (get_app_from_key(key) == appid)
        iter = correct_tag_array[index]->erase(iter);
      else
        iter++;
    }
  }
  tlb_state_tracker[appid] = 0;
}



/* MASK Implementation */
void tlb_tag_array::mask_token_reassignment()
{

  /* make this a parameter? */
  float delta = 0.1;

  for (int n = 0; n < ConfigOptions::n_apps; n++)
  {
    unsigned appID = App::get_app_id(n);
    App *app = App::get_app(appID);
    if (app->epoch_accesses == 0)
      break;
    app->epoch_previous2_miss_rate = app->epoch_previous_miss_rate;
    app->epoch_previous_miss_rate =
        (float)(app->epoch_miss /
                (float)(app->epoch_miss + app->epoch_hit + app->epoch_bypass_hit));
    app->epoch_accesses = 0;
    app->epoch_hit = 0;
    app->epoch_miss = 0;
    app->epoch_bypass_hit = 0;

    if (app->epoch_previous_miss_rate > app->epoch_previous2_miss_rate * 1.02)
    {
      /* decrease number of tokens */
      app->tokens = app->tokens - delta;
      if (app->tokens <= 0.0)
        app->tokens = 0.0;
    }
    else if (app->epoch_previous_miss_rate < app->epoch_previous2_miss_rate * 0.98)
    {
      /* increase number of tokens */
      app->tokens = app->tokens + delta;
      if (app->tokens >= 1.0)
        app->tokens = 1.0;
    }
    else
    {
      /* dont do anything */
      app->tokens = app->tokens;
    }

    unsigned total_warp_count = 0;
    unsigned handout_tokens = 0;
    /* 4000 because MASK */
    for (int i = 0; i < 4000; i++)
    {
      if (app->wid_epoch_accesses[i] == 0)
      {
        ;
      }
      else
      {
        total_warp_count++;
      }
      app->wid_epoch_accesses[i] = 0;
    }
    handout_tokens = app->tokens * total_warp_count;

    /* Give out tokens */
    for (int i = 0; i < 100; i++)
    {
      for (int s = 0; s < ConfigOptions::n_apps; s++)
      {
        unsigned global_warp_id = s * 100 + i;
        app->wid_tokens[global_warp_id] = false;
        if (app->wid_epoch_accesses[global_warp_id] == 0)
        {
          ;
        }
        else
        {
          app->wid_tokens[global_warp_id] = true;
          handout_tokens--;
          if (handout_tokens <= 0)
            break;
        }
      }
      if (handout_tokens <= 0)
        break;
    }
  }
}

/* bool tlb_tag_array::mask_tlb_fill_bypass_check(mem_fetch* mf) */
/* { */
/*   unsigned global_warp_id = mf->get_sid() * 100 + mf->get_wid(); */
/*   unsigned appID = mf->get_appID(); */
/*   App* app = App::get_app(appID); */
/*   return app->wid_tokens[global_warp_id]; */
/* } */

/* MASK: L2 TLB only */
void tlb_tag_array::mask_tlb_fill(new_addr_type addr, mem_fetch *mf)
{
  tlb_tag_array *l1_t;
  new_addr_type key =
      get_key(addr, mf->get_appID());
  bool fill_into_l2 = false;
  m_mshrs.mark_ready(key);
  page_walk_tracker.erase(key);
  while (m_mshrs.access_ready())
  {
    mem_fetch *mshr_f = m_mshrs.next_access();
    l1_t = mshr_f->get_tlb();
    assert(l1_t->m_mshrs.probe(key));
    l1_t->fill(addr, mf);
    unsigned global_warp_id = mshr_f->get_sid() * 100 + mshr_f->get_wid();
    unsigned appID = mshr_f->get_appID();
    App *app = App::get_app(appID);
    fill_into_l2 |= app->wid_tokens[global_warp_id];
  }
  if (fill_into_l2)
    fill(mf->get_addr(), mf->get_appID(), mf);
  else
    fill_bypass_cache(mf->get_addr(), mf->get_appID(), mf);
}
