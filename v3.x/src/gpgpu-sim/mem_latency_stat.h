// Copyright (c) 2009-2011, Tor M. Aamodt
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

#ifndef MEM_LATENCY_STAT_H
#define MEM_LATENCY_STAT_H

#include <stdio.h>
#include <zlib.h>
#include <map>

extern uint64_t page_walks_finished;
extern uint64_t page_walk_total_latency;

class memory_stats_t
{
   const static uint64_t stat_arr_size = 10;
   double div(uint64_t, uint64_t);
   void print_stat(const char *, uint64_t);
   void print_app_stat(const char *, uint64_t, uint64_t *, uint64_t, uint64_t *);
   void newline(void);

public:
   memory_stats_t(unsigned n_shader,
                  const struct shader_core_config *shader_config,
                  const struct memory_config *mem_config);

   void init();
   uint64_t memlatstat_done(class mem_fetch *mf);
   void memlatstat_read_done(class mem_fetch *mf);
   void memlatstat_dram_access(class mem_fetch *mf);
   void memlatstat_icnt2mem_pop(class mem_fetch *mf);
   void memlatstat_lat_pw();
   void memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk);

   void visualizer_print(gzFile visualizer_file);
   //new
   void memlatstat_print_file(unsigned n_mem, unsigned gpu_mem_n_bk, FILE *fout);
   //int get_latency(int s_id);

   //pratheek
   void page_walk_done_stat_update(mem_fetch *mf);
   void print_essential(FILE *fout);

   unsigned m_n_shader;

   const struct shader_core_config *m_shader_config;
   const struct memory_config *m_memory_config;

   uint64_t max_mrq_latency;
   uint64_t max_dq_latency;
   uint64_t max_mf_latency;
   uint64_t tlb_max_mf_latency;
   uint64_t max_icnt2mem_latency;
   uint64_t max_icnt2sh_latency;
   uint64_t mrq_lat_table[32];
   uint64_t dq_lat_table[32];
   uint64_t mf_lat_table[32];
   uint64_t icnt2mem_lat_table[24];
   uint64_t icnt2sh_lat_table[24];
   uint64_t mf_lat_pw_table[32]; //table storing values of mf latency Per Window

   uint64_t mf_total_num_lat_pw = 0;
   uint64_t tlb_mf_total_num_lat_pw = 0;
   uint64_t data_mf_total_num_lat_pw = 0;

   uint64_t dram_app_switch;
   uint64_t max_warps;

   uint64_t mf_total_tot_lat_pw = 0;
   uint64_t tlb_mf_total_tot_lat_pw = 0;
   uint64_t data_mf_total_tot_lat_pw = 0;
   uint64_t mf_total_total_lat = 0;
   uint64_t tlb_mf_total_total_lat = 0;
   uint64_t data_mf_total_total_lat = 0;

   uint64_t high_prio_queue_count;

   uint64_t coalesced_tried;
   uint64_t coalesced_succeed;
   uint64_t coalesced_noinval_succeed;
   uint64_t coalesced_fail;

   //Number of pages not being used when being coalesced
   uint64_t max_bloat;
   uint64_t num_coalesce;
   uint64_t pt_space_size;

   uint64_t tlb_bypassed;
   uint64_t tlb_bypassed_level[stat_arr_size];
   uint64_t tlb_bypassed_core[200];
   uint64_t tlb_level_accesses[stat_arr_size];
   uint64_t tlb_level_hits[stat_arr_size];
   uint64_t tlb_level_misses[stat_arr_size];
   uint64_t tlb_level_fails[stat_arr_size];

   uint64_t l2_cache_accesses = 0;
   uint64_t l2_cache_hits = 0;
   uint64_t l2_cache_misses = 0;

   float TLBL1_sharer_avg[200];
   uint64_t TLBL1_total_unique_addr[200];
   float TLBL1_sharer_var[200];
   uint64_t TLBL1_sharer_max[200];
   uint64_t TLBL1_sharer_min[200];

   uint64_t TLB_bypass_cache_flush_stalled;
   uint64_t TLB_L1_flush_stalled[200];
   uint64_t TLB_L2_flush_stalled;

   float TLBL2_sharer_avg;
   uint64_t TLBL2_total_unique_addr;
   float TLBL2_sharer_var;
   uint64_t TLBL2_sharer_max;
   uint64_t TLBL2_sharer_min;

   uint64_t **mf_total_lat_table; //mf latency sums[dram chip id][bank id]
   uint64_t **mf_max_lat_table;   //mf latency sums[dram chip id][bank id]

   uint64_t total_num_mfs = 0;
   uint64_t tlb_total_num_mfs = 0;
   uint64_t data_total_num_mfs = 0;

   uint64_t ***bankwrites;           //bankwrites[shader id][dram chip id][bank id]
   uint64_t ***bankreads;            //bankreads[shader id][dram chip id][bank id]
   uint64_t **totalbankblocked;      //number of cycles banks are blocked [dram chip id][bank id]
   uint64_t **totalbankwrites;       //bankwrites[dram chip id][bank id]
   uint64_t **totalbankreads;        //bankreads[dram chip id][bank id]
   uint64_t **totalbankaccesses;     //bankaccesses[dram chip id][bank id]
   uint64_t *num_MCBs_accessed;      //tracks how many memory controllers are accessed whenever any thread in a warp misses in cache
   uint64_t *position_of_mrq_chosen; //position of mrq in m_queue chosen

   uint64_t ***mem_access_type_stats; // dram access type classification

   // TLB-related

   uint64_t totalL1TLBMissesAll;
   uint64_t totalL2TLBMissesAll;

   uint64_t sched_from_normal_prio;
   uint64_t sched_from_high_prio;
   uint64_t sched_from_special_prio;
   uint64_t DRAM_normal_prio;
   uint64_t DRAM_high_prio;
   uint64_t DRAM_special_prio;
   uint64_t drain_reset;
   uint64_t total_combo;

   uint64_t **totalL1TLBMisses;           //totalL1TLBMisses[shader id][app id]
   uint64_t **totalL2TLBMisses;           //totalL2TLBMisses[shader id][app id]
   uint64_t **totalTLBMissesCausedAccess; //totalTLBMissesCausedAccess[shader id][app id]

   // L2 cache stats
   uint64_t *L2_cbtoL2length;
   uint64_t *L2_cbtoL2writelength;
   uint64_t *L2_L2tocblength;
   uint64_t *L2_dramtoL2length;
   uint64_t *L2_dramtoL2writelength;
   uint64_t *L2_L2todramlength;

   // DRAM access row locality stats
   uint64_t **concurrent_row_access; //concurrent_row_access[dram chip id][bank id]

   uint64_t **num_activates;
   uint64_t **row_access;

   uint64_t **num_activates_w;
   uint64_t **row_access_w;

   uint64_t **max_conc_access2samerow; //max_conc_access2samerow[dram chip id][bank id]
   uint64_t **max_servicetime2samerow; //max_servicetime2samerow[dram chip id][bank id]

   // Power stats
   uint64_t total_n_access;
   uint64_t total_n_reads;
   uint64_t total_n_writes;

   //pratheek: required statistics
   uint64_t l2_tlb_tot_hits, l2_tlb_tot_misses, l2_tlb_tot_accesses,
            l2_tlb_tot_mshr_hits, l2_tlb_tot_mshr_fails,
            l2_tlb_tot_backpressure_fails,
            l2_tlb_tot_backpressure_stalls;
   uint64_t l2_tlb_app_hits[stat_arr_size], l2_tlb_app_misses[stat_arr_size],
   l2_tlb_app_accesses[stat_arr_size],
   l2_tlb_app_mshr_hits[stat_arr_size], l2_tlb_app_mshr_fails[stat_arr_size],
   l2_tlb_app_backpressure_fails[stat_arr_size],
   l2_tlb_app_backpressure_stalls[stat_arr_size];

   uint64_t pwq_tot_lat; //Neha: record the amount of time spent by a request in the page walk queue
   uint64_t pwq_app_lat[stat_arr_size];
   uint64_t pwq_app_pw_stolen[3][stat_arr_size];
   uint64_t pwq_app_intf[stat_arr_size];

   //page walk statistics
   uint64_t pw_tot_lat, pw_tot_num;

   //per app page walk stats
   uint64_t pw_app_num[stat_arr_size];
   uint64_t pw_app_lat[stat_arr_size];
   // uint64_t per_app_total_cycle_interference[stat_arr_size];

   // double per_app_avg_interference[stat_arr_size];

   double avg_time_on_pw_queue;

   uint64_t pwc_tot_accesses, pwc_tot_hits, pwc_tot_misses;
   uint64_t pwc_tot_addr_lvl_accesses[stat_arr_size], pwc_tot_addr_lvl_hits[stat_arr_size], pwc_tot_addr_lvl_misses[stat_arr_size];
   uint64_t pwc_app_accesses[stat_arr_size], pwc_app_hits[stat_arr_size], pwc_app_misses[stat_arr_size];
   uint64_t pwc_app_addr_lvl_accesses[stat_arr_size][stat_arr_size], pwc_app_addr_lvl_hits[stat_arr_size][stat_arr_size], pwc_app_addr_lvl_misses[stat_arr_size][stat_arr_size];

   //inter app walker interference
   uint64_t page_walk_queue_insert_self[stat_arr_size];
   uint64_t page_walk_queue_insert_other[stat_arr_size];
   uint64_t page_walker_service_self[stat_arr_size];
   uint64_t page_walker_service_other[stat_arr_size];

   uint64_t local_queue_interference[stat_arr_size];
   uint64_t stealing_from_self[stat_arr_size];
   uint64_t stealing_from_other[stat_arr_size];

   uint64_t stealing_activated_dwsp[stat_arr_size];
   uint64_t occupancy_threshold_triggered[4][stat_arr_size];
   uint64_t occupancy_difference[10];

   uint64_t tlb_occupancy_sum[stat_arr_size];
   uint64_t tlb_occupancy_end[stat_arr_size];
   uint64_t tlb_fills[stat_arr_size];
   uint64_t pwc_occupancy_sum[stat_arr_size];
   uint64_t pwc_occupancy_end[stat_arr_size];

   uint64_t sum_num_page_walkers_assigned[stat_arr_size];
   uint64_t end_num_page_walkers_assigned[stat_arr_size];
};

#endif /*MEM_LATENCY_STAT_H*/
