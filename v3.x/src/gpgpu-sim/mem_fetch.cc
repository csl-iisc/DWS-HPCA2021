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

#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"
#include "gpu-sim.h"

#include "pagewalk.h"

unsigned mem_fetch::sm_next_mf_request_uid=1;

extern int gpu_sms;

int page_walks_returning_depth[5] = {0, 0, 0, 0, 0};
int page_walks_generated = 0;
int tlb_requests_generated = 0;
int pw_cache_hits = 0;
int pw_cache_latency_queue_pushes = 0;
int num_page_walk_active = 0;
int last_page_walk_return = 0;

mem_fetch::mem_fetch( mem_fetch * mf )
{

  page_walks_generated++;

  pwcache_hit = false;
  pwcache_done = false;

  m_request_uid = sm_next_mf_request_uid++;

  m_access = mf->get_access();
  m_tlb_depth_count = mf->get_tlb_depth_count() + 1;
  page_walker = mf->page_walker;
  m_tlb = mf->get_tlb();
  new_addr_type new_addr = (mf->get_appID() << 48) | m_tlb->get_tlbreq_addr(mf);
  set_addr(new_addr);
  m_inst = mf->get_inst();
  assert( mf->get_wid() == m_inst.warp_id() );
  m_data_size = mf->get_data_size();
  m_ctrl_size = mf->get_ctrl_size();
  m_sid = mf->get_sid();
  m_tpc = mf->get_tpc();
  m_wid = mf->get_wid();

  // The address mapping starts here
  // Assumption --> mf contains appID and actual address already

  m_appID = mf->get_appID();
  m_tlb_related_req = true;
  m_tlb_miss = false;
  m_original_addr = mf->get_original_addr();
  m_mem_config = mf->get_mem_config();

  set_page_fault( \
      m_mem_config->m_address_mapping.addrdec_tlx(m_access.get_addr(),
        &m_raw_addr, m_appID, m_tlb_depth_count, !m_access.is_write()));

  m_partition_addr =
    m_mem_config->m_address_mapping.partition_address(m_access.get_addr(),
        m_appID, m_tlb_depth_count,!m_access.is_write());

  m_type = READ_REQUEST;
  m_timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = gpu_sim_cycle + gpu_tot_sim_cycle;
  icnt_flit_size = m_mem_config->icnt_flit_size;
  beenThroughL1 = false; 

  if(mf->get_cache()==NULL)
  {
    printf("Rachata-debug: Whoaaa,\
        cache is not set, cannot probe the cache!!!\n");}
  else 
  {
    m_cache = mf->get_cache();
    m_core_id = m_cache->get_core_id();
  }

  accum_dram_access = 0;
  m_parent_tlb_request = mf;
  bypass_L2 = false;
  m_page_fault = false;
  isL2TLBMiss = false;
  page_walker = NULL;

}

mem_fetch::mem_fetch( const mem_access_t &access, const warp_inst_t *inst,
    unsigned ctrl_size, unsigned wid, unsigned sid, unsigned tpc, 
    const class memory_config *config )
{

  m_request_uid = sm_next_mf_request_uid++;
  m_access = access;
  if( inst ) { 
    m_inst = *inst;
    assert( wid == m_inst.warp_id() );
  }
  m_data_size = access.get_size();
  m_ctrl_size = ctrl_size;
  m_sid = sid;
  m_tpc = tpc;
  m_wid = wid;
  m_tlb = NULL; // By default, this should be null
  m_type = m_access.is_write()?WRITE_REQUEST:READ_REQUEST;
  m_timestamp = gpu_sim_cycle + gpu_tot_sim_cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = gpu_sim_cycle + gpu_tot_sim_cycle;
  m_mem_config = config;
  icnt_flit_size = config->icnt_flit_size;

  m_original_addr = m_access.get_addr();

  m_tlb_related_req = false;

  m_cache = NULL;

  /* FROM ORIGINAL MASK CODE START*/
  // After a longgg time, for some reason sid becomes -1 ... This causes a
  // segfault for App object because the index is out of bound
  // Got it ... Seems to come from mem_fetch allocator if it is created by
  // the l2cache. For some reason it just don't want to pass the proper appID
  /* FROM ORIGINAL MASK CODE STOP*/
  m_appID = App::get_app_id_from_sm(sid);

  set_page_fault(config->m_address_mapping.addrdec_tlx(access.get_addr(),&m_raw_addr, m_appID, 0,!m_access.is_write()));
  m_partition_addr = config->m_address_mapping.partition_address(access.get_addr(), m_appID, 0,!m_access.is_write());

  m_tlb_depth_count = 0;
  //set_tlb_miss(false);
  m_tlb_miss = false;

  m_page_fault = false;


  m_parent_tlb_request = NULL;

  accum_dram_access = 0;

  // Rachata: Only set from the L2 TLB, if this is a TLB miss at L2 TLB
  isL2TLBMiss = false;

  bypass_L2 = false;
  pwcache_hit = false;
  pwcache_done = false;
  beenThroughL1 = false;

  been_through_tlb = false;

  page_walker = NULL;
}

mem_fetch::~mem_fetch()
{
  m_status = MEM_FETCH_DELETED;
}

#define MF_TUP_BEGIN(X) static const char* Status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) };
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

void mem_fetch::set_page_fault(bool val){
  m_page_fault = val;
}

// Check if the request is a write, This is rewritten to make sure any TLB-related req returns as a read.
bool mem_fetch::is_write() {
  if(m_tlb_depth_count > 0) return false; 
  else  return m_access.is_write();
}

// Check if the request is a write, This is rewritten to make sure any TLB-related req returns as a read. For some stupid reasons gpgpu_sim has both is_write and get_is_write ...
bool mem_fetch::get_is_write() const { 
  if(m_tlb_depth_count > 0) return false; 
  else  return m_access.is_write();
}

void mem_fetch::set_cache(data_cache * cache)
{
  m_cache = cache;
  m_core_id = m_cache->get_core_id();
}

data_cache * mem_fetch::get_cache(){return m_cache;}

int mem_fetch::get_core()
{
  return m_core_id;
}

unsigned mem_fetch::get_subarray(){
  return m_raw_addr.subarray;
}

bool mem_fetch::check_bypass(mem_fetch * mf)
{
  if(m_mem_config->tlb_bypass_enabled == 0)
  {
    return false;
  }

  if(m_mem_config->tlb_bypass_enabled == 1 &&
      ( mf->get_tlb_depth_count() >= m_mem_config->tlb_bypass_level))
  {
    return true;
  }
  else if(m_mem_config->tlb_bypass_enabled == 2)
  {
        if( (float)(mf->get_tlb()->get_shared_tlb()->get_mem_stat()->tlb_level_hits[mf->get_tlb_depth_count()])/  (float)(mf->get_tlb()->get_shared_tlb()->get_mem_stat()->tlb_level_accesses[mf->get_tlb_depth_count()]) /*TLB level hit rate*/ 
         <  (float)(mf->get_tlb()->get_shared_tlb()->get_mem_stat()->l2_cache_hits)/  (float)(mf->get_tlb()->get_shared_tlb()->get_mem_stat()->l2_cache_accesses) /*L2 data cache hit rate*/ ) //Can get from l2cache.cc line 770, or just collect it again
            return true;
        else
            return false;
  }
  else
  {
    return false;
  }

}

//Rachata: Note: when the request is serviced in DRAM, invoke this routine
//again on the parent
//This code is on l2cache.cc in dram_cycle(); 
//This routine is invoked only with a tlb request got its data from
//DRAM (at the DRAM return queue pop)

void mem_fetch::done_tlb_req(mem_fetch * mf)
{
  last_page_walk_return = gpu_sim_cycle;
  page_walks_returning_depth[mf->get_tlb_depth_count()]++;

  struct tlb_tag_array *l2tlb = mf->get_tlb()->get_shared_tlb();
  memory_stats_t* mem_stats = l2tlb->get_mem_stat();
  PageWalker *page_walker = mf->page_walker;

  /* If it is an page table access */
  if(mf->get_parent_tlb_request()!=NULL)
  {

    /* If PW cache hit */
    if(mf->pwcache_hit && !mf->pwcache_done)
    {
      pw_cache_latency_queue_pushes++;
      /* Put this request in the latency queue for a PW cache hit */
      mf->get_tlb()->get_shared_tlb()->page_walk_subsystem->
        page_walk_cache_enqueue(mf);
      return;
    }
    else if(mf->pwcache_hit && mf->pwcache_done)
    {
      pw_cache_hits++;
      /* Pratheek: attatch the page walker to the parent */
      mf->get_parent_tlb_request()->page_walker = mf->page_walker;
      done_tlb_req(mf->get_parent_tlb_request());

      mf->get_tlb()->get_mem_stat()->memlatstat_read_done(mf);

      delete mf;
      return;
    }

    mf->get_parent_tlb_request()->accum_dram_access =
      mf->accum_dram_access+1;

    mf->bypass_L2 = mf->check_bypass(mf);
    mf->get_parent_tlb_request()->page_walker = mf->page_walker;
    /* Send the actual (original) memory request to DRAM */
    m_cache->add_tlb_miss_to_queue(mf);

  }
  /* If the memory fetch is done */
  else
  {
    page_walker->page_walk_return(mf);
  }
}

void mem_fetch::print( FILE *fp, bool print_inst ) const
{
  if(this == NULL)
  {
    fprintf(fp," <NULL mem_fetch pointer>\n");
    return;
  }

  fprintf(fp,"  mf: uid=%6u, sid%02u:w%02u, part=%u, ",
      m_request_uid, m_sid, m_wid, m_raw_addr.chip );

  m_access.print(fp);

  if((unsigned)m_status < NUM_MEM_REQ_STAT) 
  {
    fprintf(fp," status = %s (%llu), ",
        Status_str[m_status], m_status_change );
  }
  else
  {
    fprintf(fp," status = %u??? (%llu), ", m_status, m_status_change );
  }

  if(!m_inst.empty() && print_inst)
    m_inst.print(fp);
  else
    fprintf(fp,"\n");
}

void mem_fetch::set_status(enum mem_fetch_status status,
    unsigned long long cycle) 
{
  m_status = status;
  m_status_change = cycle;
}

bool mem_fetch::isatomic() const
{
   if( m_inst.empty() ) return false;
   return m_inst.isatomic();
}

void mem_fetch::do_atomic()
{
    m_inst.do_atomic( m_access.get_warp_mask() );
}

bool mem_fetch::istexture() const
{
    if( m_inst.empty() ) return false;
    return m_inst.space.get_type() == tex_space;
}

bool mem_fetch::isconst() const
{ 
    if( m_inst.empty() )
      return false;
    return (m_inst.space.get_type() == const_space) ||
      (m_inst.space.get_type() == param_space_kernel);
}

/* Returns number of flits traversing interconnect. */
/* simt_to_mem specifies the direction */
unsigned mem_fetch::get_num_flits(bool simt_to_mem)
{
	unsigned sz=0;
	// If atomic, write going to memory, or read coming back from memory,
  // size = ctrl + data. Else, only ctrl
	if( isatomic() || (simt_to_mem && get_is_write()) ||
      !(simt_to_mem || get_is_write()) )
		sz = size();
	else
		sz = get_ctrl_size();

	return (sz/icnt_flit_size) + ( (sz % icnt_flit_size)? 1:0);
}
