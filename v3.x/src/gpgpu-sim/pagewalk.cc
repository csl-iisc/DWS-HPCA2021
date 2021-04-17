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

#include "pagewalk.h"
#include "mem_latency_stat.h" /* Neha: I'm not sure where this header should be included tbh */

/* Page Walk Sub System methods */

PageWalkSubsystem::PageWalkSubsystem(tlb_tag_array *tlb, mmu *page_manager,
                                     const memory_config *config, memory_stats_t *mem_stat)
{

  this->config = config;
  this->mem_stat = mem_stat;
  L2_TLB = tlb;
  this->page_manager = page_manager;

  /* This order of construction needs to be maintained for correctness,
   * else NULL pointers may creep in and creep you out.
   * */
  page_walk_cache = new PageWalkCache(this, config, mem_stat);
  page_walk_queue = new PageWalkQueue(this, config, mem_stat);
  num_walkers = config->concurrent_page_walks;
  for (unsigned i = 0; i < num_walkers; i++)
  {
    page_walkers.push_back(new PageWalker(this, config, mem_stat, i));
  }
  for (int n = 0; n < ConfigOptions::n_apps; n++)
  {
    unsigned appid = App::get_app_id(n);
    current_allocation[appid] = 0;
    occupancy_in_this_dwsp_epoch_per_app[appid] = 0;
    number_of_enqueues_in_this_dwsp_epoch_per_app[appid] = 0;
    number_of_walks_serviced_by_stealing[appid] = 0;
    page_walk_in_this_dwsp_epoch_per_app[appid] = 0;
  }
  dwsp_occupancy_threshold = 1.0;
  dwsp_epoch_number = 0;
  page_walk_in_this_dwsp_epoch = 0;
}

bool PageWalkSubsystem::enqueue(mem_fetch *mf)
{
  if (page_walk_queue->enqueue(mf))
  {
    return true;
  }
  else
  {
    if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      config->vm_config == VM_MASK_PLUS_DWS ||
      config->vm_config == VM_DWS_PER_APP_TLB ||
        config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
    {
      L2_TLB->stall_app[mf->get_appID()] = true;
    }
    else
    {
      L2_TLB->stall = true;
    }
    return false;
  }
}

mem_fetch *PageWalkSubsystem::dequeue(PageWalker *page_walker)
{
  mem_fetch *page_walk = NULL;
  page_walk = page_walk_queue->dequeue(page_walker);
  if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      config->vm_config == VM_MASK_PLUS_DWS ||
      config->vm_config == VM_DWS_PER_APP_TLB ||
      config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
  {
    if (page_walk)
      L2_TLB->stall_app[page_walk->get_appID()] = false;
  }
  else
  {
    L2_TLB->stall = false;
  }

  return page_walk;
}

unsigned PageWalkSubsystem::get_page_walk_queue_enqueue(mem_fetch *mf)
{
  return page_walk_queue->get_page_walk_queue_enqueue(mf->get_appID());
}

unsigned PageWalkSubsystem::get_num_walkers()
{
  return num_walkers;
}

struct tlb_tag_array *PageWalkSubsystem::get_tlb()
{
  return L2_TLB;
}

PageWalker *PageWalkSubsystem::get_idle_page_walker()
{

  /* Assuming Baseline 
   * Needs to incorporate partition information for DWS
   * */
  std::vector<PageWalker *>::iterator it;
  for (it = page_walkers.begin(); it != page_walkers.end(); it++)
  {
    if ((*it)->get_current() == NULL)
    {
      return (*it);
    }
  }
  return NULL;
}

void PageWalkSubsystem::service_page_walk_cache_queue()
{
  page_walk_cache->service_latency_queue();
}

void PageWalkSubsystem::page_walk_cache_enqueue(mem_fetch *mf)
{
  page_walk_cache->enqueue(mf);
}

void PageWalkSubsystem::dwsp_epoch_tracking(mem_fetch* mf)
{
  page_walk_in_this_dwsp_epoch++;
  page_walk_in_this_dwsp_epoch_per_app[mf->get_appID()]++;
  if(page_walk_in_this_dwsp_epoch == config->dwsp_epoch_length)
  {
    /* do calculate the threshold  */
    /* TODO: divide by zero handling */
    double app1_by_app2;
    double app2_by_app1;
    if(page_walk_in_this_dwsp_epoch_per_app[2] == 0)
      app1_by_app2 = config->dwsp_epoch_length;
    else
      app1_by_app2 = (double)page_walk_in_this_dwsp_epoch_per_app[1] /
        (double)page_walk_in_this_dwsp_epoch_per_app[2];
    if(page_walk_in_this_dwsp_epoch_per_app[1] == 0)
      app2_by_app1 = config->dwsp_epoch_length;
    else
      app2_by_app1 = (double)page_walk_in_this_dwsp_epoch_per_app[2] /
        (double)page_walk_in_this_dwsp_epoch_per_app[1];

    double diff_walks = app1_by_app2 > app2_by_app1 ? app1_by_app2 : app2_by_app1;

#if 1
    printf("\n");
    printf("epoch over: number of walks serviced by stealng in epoch\n");
    for (int n = 0; n < ConfigOptions::n_apps; n++)
    {
      printf("%d\t", number_of_walks_serviced_by_stealing[App::get_app_id(n)]);
    }
    printf("\n");
    printf("epoch over: stats: number of walks in epoch\n");
    for (int n = 0; n < ConfigOptions::n_apps; n++)
    {
      printf("%d\t", page_walk_in_this_dwsp_epoch_per_app[App::get_app_id(n)]);
    }
    printf("\n");
    printf("epoch over: stats: average occupancy in epoch\n");
    for (int n = 0; n < ConfigOptions::n_apps; n++)
    {
      if(number_of_enqueues_in_this_dwsp_epoch_per_app[App::get_app_id(n)] == 0)
        printf("0.0\t");
      else
        printf("%lf\t", occupancy_in_this_dwsp_epoch_per_app[App::get_app_id(n)]/
            number_of_enqueues_in_this_dwsp_epoch_per_app[App::get_app_id(n)]);
    }
    printf("\n");
    printf("epoch over: stats: app1_by_app2 = %lf\n", app1_by_app2);
    printf("epoch over: stats: app2_by_app1 = %lf\n", app2_by_app1);
    printf("epoch over: diff_walks:         = %lf\n", diff_walks);
#endif

    /* the long ladder */
    if(diff_walks < config->diff_walks_threshold_1)
      dwsp_occupancy_threshold = config->dwsp_occupancy_threshold_1;
    else if(diff_walks < config->diff_walks_threshold_2)
      dwsp_occupancy_threshold = config->dwsp_occupancy_threshold_2;
    else if(diff_walks < config->diff_walks_threshold_3)
      dwsp_occupancy_threshold = config->dwsp_occupancy_threshold_3;
    else if(diff_walks < config->diff_walks_threshold_4)
      dwsp_occupancy_threshold = config->dwsp_occupancy_threshold_4;
    else 
      dwsp_occupancy_threshold = config->dwsp_occupancy_threshold_5;

#if 1
    printf("epoch over: chosen diff thres:  = %lf\n", dwsp_occupancy_threshold);
#endif
    

    /* reset the number of page walks per epoch */
    page_walk_in_this_dwsp_epoch = 0;
    for (int n = 0; n < ConfigOptions::n_apps; n++)
    {
      page_walk_in_this_dwsp_epoch_per_app[App::get_app_id(n)] = 0;
      occupancy_in_this_dwsp_epoch_per_app[App::get_app_id(n)] = 0;
      number_of_walks_serviced_by_stealing[App::get_app_id(n)] = 0;
      number_of_enqueues_in_this_dwsp_epoch_per_app[App::get_app_id(n)] = 0;
    }
  }
}

/* Page Walker methods */

PageWalker::PageWalker(PageWalkSubsystem *page_walk_subsystem,
                       const memory_config *config, memory_stats_t *mem_stat, unsigned id)
{

  this->page_walk_subsystem = page_walk_subsystem;
  this->config = config;
  this->mem_stat = mem_stat;
  this->id = id;
  this->previous_translation_finish_cycle = 0;
  current = NULL;
  appid = -1;
  page_walk_cache = page_walk_subsystem->page_walk_cache;
  if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      config->vm_config == VM_MASK_PLUS_DWS ||
      config->vm_config == VM_DWS_PER_APP_TLB ||
      config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
  {
    appid = App::get_app_id(id % ConfigOptions::n_apps);
    last_serviced_stolen = config->stealing_freq_threshold;
    /* last_serviced_stolen = (last_serviced_stolen + 1 ) % 4; */
  }
  current_is_stolen = false;
}

void PageWalker::initiate_page_walk(mem_fetch *mf)
{

  /* Initiate the page walk using the page walk mechanishm */
  mem_fetch *pw = pt_walk(mf);
  pw->page_walker = this;
  this->current = mf;
  page_walk_subsystem->current_allocation[mf->get_appID()]++;
  pw->done_tlb_req(pw);
}

mem_fetch *PageWalker::get_current()
{
  return current;
}

void PageWalkSubsystem::stealing_latency_pause(PageWalker* page_walker)
{
  page_walker->previous_translation_finish_cycle = gpu_sim_cycle;
  this->stealing_latency_queue.push_back(page_walker);
}

void PageWalkSubsystem::stealing_latency_resume()
{
  while(PageWalker* page_walker = stealing_latency_queue.front())
  {
    if(page_walker->previous_translation_finish_cycle +
        config->stealing_latency_enabled > gpu_sim_cycle)
    {
      break;
    }

    page_walker->current = NULL;
    mem_fetch* mf = dequeue(page_walker);

    if (mf)
    {
      page_walker->initiate_page_walk(mf);
    }
    stealing_latency_queue.pop_front();
  }
}

bool PageWalker::page_walk_return(mem_fetch *mf)
{

  /* Part 1: Fill into TLB and finish off the page walk */
  mf->been_through_tlb = true;
  mf->set_tlb_miss(false);
  if(config->vm_config == VM_MASK || config->vm_config == VM_MASK_PLUS_DWS)
  {
    mf->get_tlb()->m_shared_tlb->mask_tlb_fill(mf->get_addr(), mf);
  }
  else
  {
    mf->get_tlb()->l2_fill(mf->get_addr(), mf->get_appID(), mf);
    mf->get_tlb()->m_shared_tlb->fill_into_l1_tlb(mf->get_addr(), mf);
  }

  memory_stats_t *mem_stat = this->mem_stat;
  mem_stat->pw_tot_num++;
  unsigned cur_pw_lat = (gpu_sim_cycle + gpu_tot_sim_cycle) - mf->get_timestamp();
  mem_stat->pw_tot_lat += cur_pw_lat;
  unsigned app_id = mf->get_appID();
  mem_stat->pw_app_num[app_id]++;
  mem_stat->pw_app_lat[app_id] += cur_pw_lat;
  page_walk_subsystem->current_allocation[app_id]--;

  /* simulate extra latency if servicing a stolen walk */
  if(config->stealing_latency_enabled && current_is_stolen)
  {
    page_walk_subsystem->stealing_latency_pause(this);
    return false;
  }

  this->current = NULL;
  /* Part 2: Start a new page walk if there are outstanding page walks to 
   * service */

  mf = page_walk_subsystem->dequeue(this);

  if (mf)
  {
    initiate_page_walk(mf);
  }
  return true;
}

mem_fetch *PageWalker::pt_walk(mem_fetch *mf)
{
  /* Setup dependent memory requests for DRAM to be serviced for a TLB miss */
  if (config->tlb_fixed_latency_enabled)
  {
    page_walk_subsystem->L2_TLB->put_mf_to_static_queue(mf);
    return NULL;
  }

  /* Done with setting up the last request in the PT walk routine */
  if (mf->get_tlb_depth_count() >= config->tlb_levels)
  {
    return mf;
  }
  else
  {
    mem_fetch *child;
    /* Set a new mem_fetch for the next level subroutine */
    child = new mem_fetch(mf);

    if (config->pw_cache_enable)
      probe_pw_cache(child);

    mf->set_child_tlb_request(child);
    /* Then, continue performing the page table walk for
     * the next level of TLB access */
    return pt_walk(mf->get_child_tlb_request());
  }
}

void PageWalker::probe_pw_cache(mem_fetch *mf)
{
  assert(mf->get_tlb_depth_count() != 0);
  if (mf->get_tlb_depth_count() == 0 || mf->get_tlb_depth_count() == 1)
  {
    mf->pwcache_hit = false;
  }
  else
  {
    memory_stats_t *mem_stat = this->mem_stat;
    mem_stat->pwc_tot_accesses++;
    unsigned app_id = mf->get_appID();
    unsigned addr_lvl = mf->get_tlb_depth_count();
    mem_stat->pwc_app_accesses[app_id]++;
    mem_stat->pwc_tot_addr_lvl_accesses[addr_lvl]++;
    mem_stat->pwc_app_addr_lvl_accesses[app_id][addr_lvl]++;
    if (page_walk_cache->access(mf))
    {
      mf->pwcache_hit = true;
      mem_stat->pwc_tot_hits++;
      mem_stat->pwc_app_hits[app_id]++;
      mem_stat->pwc_tot_addr_lvl_hits[addr_lvl]++;
      mem_stat->pwc_app_addr_lvl_hits[app_id][addr_lvl]++;
    }
    else
    {
      mf->pwcache_hit = false;
      mem_stat->pwc_tot_misses++;
      mem_stat->pwc_app_misses[app_id]++;
      mem_stat->pwc_tot_addr_lvl_misses[addr_lvl]++;
      mem_stat->pwc_app_addr_lvl_misses[app_id][addr_lvl]++;
    }
  }
}

void PageWalker::print()
{
  printf("PageWalker ID    : %2d\n", id);
  printf("Current mem_fetch: %p\n", current);
  printf("Original Addr    : %lx\n", current->get_original_addr());
  printf("\n");
}

/* Page Walk Cache methods */

PageWalkCacheImpl::PageWalkCacheImpl(const memory_config *config, memory_stats_t *mem_stat)
{

  this->config = config;
  this->mem_stat = mem_stat;

  ports = config->pw_cache_num_ports;

  pw_cache_entries = config->tlb_pw_cache_entries;
  pw_cache_ways = config->tlb_pw_cache_ways;
  pw_cache = new std::list<new_addr_type> *[pw_cache_entries];
  for (int i = 0; i < pw_cache_entries; i++)
    pw_cache[i] = new std::list<new_addr_type>;

  pw_cache_lat_queue = new std::list<mem_fetch *>;
  pw_cache_lat_time = new std::list<unsigned long long>;
}

bool PageWalkCacheImpl::access(new_addr_type key, unsigned index)
{
  std::list<new_addr_type>::iterator findIter =
      std::find(pw_cache[index]->begin(), pw_cache[index]->end(), key);
  if (findIter != pw_cache[index]->end())
  {
    pw_cache[index]->remove(key);
    pw_cache[index]->push_front(key);
    return true;
  }
  else
  {
    fill(key, index);
    return false;
  }
}

bool PageWalkCacheImpl::fill(new_addr_type key, unsigned index)
{
  pw_cache[index]->remove(key);

  while (pw_cache[index]->size() >= pw_cache_ways)
  {
    pw_cache[index]->pop_back();
  }
  pw_cache[index]->push_front(key);
}

void PageWalkCacheImpl::service_latency_queue()
{
  int ports = 0;
  while (!pw_cache_lat_queue->empty())
  {
    unsigned long long temp = pw_cache_lat_time->front();
    if ((ports < config->pw_cache_num_ports) &&
        ((temp + config->pw_cache_latency) <
         gpu_sim_cycle + gpu_tot_sim_cycle))
    {
      mem_fetch *mf = pw_cache_lat_queue->front();

      // Remove mf from the list
      pw_cache_lat_time->pop_front();
      pw_cache_lat_queue->pop_front();

      // Finish up the current pw cache hit routine, call the next mf
      mf->pwcache_done = true;
      mf->done_tlb_req(mf);
      /* Rachata's comment: Note that the actual done_tlb_req is called in
       * mem_fetch, at this point we just need to mark pwcache as done.
       * (Otherwise this we will call parent request on both PW cache hit
       * and PW cache miss requests */

      ports++;
    }
    else
      break;
  }
}

void PageWalkCacheImpl::enqueue(mem_fetch *mf)
{
  pw_cache_lat_queue->push_front(mf);
  pw_cache_lat_time->push_front(mf->get_timestamp());
}

void PageWalkCacheImpl::print()
{
  int count = 0;
  for (int index = 0; index < pw_cache_entries; index++)
  {
    printf("%p\n", index);
    for (std::list<new_addr_type>::iterator it =
             pw_cache[index]->begin();
         it != pw_cache[index]->end(); ++it)
    {
      printf("%p\t", *it);
      count++;
    }
    printf("\n\n");
  }
  printf("entries in pw cache: %d\n", count);
}

PageWalkCache::PageWalkCache(PageWalkSubsystem *page_walk_subsystem,
                             const memory_config *config, memory_stats_t *mem_stat)
{

  this->page_walk_subsystem = page_walk_subsystem;
  this->config = config;
  this->mem_stat = mem_stat;
  this->page_manager = page_walk_subsystem->page_manager;
  pw_cache_entries = config->tlb_pw_cache_entries;
  pw_cache_ways = config->tlb_pw_cache_ways;

  // This needs to change for multiple page walk caches.
  per_app_pwc_used = false;
  pwc = new PageWalkCacheImpl(config, mem_stat);

  ports = config->pw_cache_num_ports;
  for (int n = 0; n < ConfigOptions::n_apps; n++)
  {
    unsigned appid = App::get_app_id(n);
    state_tracker[appid] = 0;
  }
}

bool PageWalkCache::access(mem_fetch *mf)
{
  new_addr_type key = get_key(mf);
  unsigned index = (mf->get_addr() >> (config->page_size)) &
                   (pw_cache_entries - 1);
  return pwc->access(key, index);
}

bool PageWalkCache::fill(mem_fetch *mf)
{
  new_addr_type key = get_key(mf);
  unsigned index = (mf->get_addr() >> (config->page_size)) &
                   (pw_cache_entries - 1);
  state_tracker[mf->get_appID()]++;
  return pwc->fill(key, index);
}

new_addr_type PageWalkCache::get_key(mem_fetch *mf)
{
  unsigned key = mf->get_original_addr();

  unsigned bitmask = page_manager->get_bitmask(mf->get_tlb_depth_count());
  key = key & bitmask;
  /* Process the offset, bitmask should be increasingly longer as depth
   * increases. Then need to shift the bitmask */

  unsigned temp_mask = bitmask;
  while ((bitmask > 0) && ((bitmask & 1) == 0))
  {
    key = key >> 1;
    bitmask = bitmask >> 1;
  }

  key = (key << 4) | mf->get_appID();

  return key;
}

void PageWalkCache::service_latency_queue()
{
  pwc->service_latency_queue();
}

void PageWalkCache::enqueue(mem_fetch *mf)
{
  pwc->enqueue(mf);
}

void PageWalkCache::print()
{
  pwc->print();
}

/* Page Walk Queue methods */

PageWalkQueue::PageWalkQueue(PageWalkSubsystem *page_walk_subsystem,
                             const memory_config *config, memory_stats_t *mem_stat)
{

  this->page_walk_subsystem = page_walk_subsystem;
  this->config = config;
  this->mem_stat = mem_stat;

  /* This needs to change for QoS provisioning of page walkers. */
  size = config->page_walk_queue_size;
  per_walker_queue_used = false;
  if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      config->vm_config == VM_MASK_PLUS_DWS ||
      config->vm_config == VM_DWS_PER_APP_TLB ||
      config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
  {
    size = config->page_walk_queue_size / config->concurrent_page_walks;
    per_walker_queue_used = true;
    for (int i = 0; i < config->concurrent_page_walks; i++)
    {
      std::list<mem_fetch *> *queue = new std::list<mem_fetch *>;
      per_walker_queue.push_back(queue);
      walker_to_app_map[i] = App::get_app_id(i % ConfigOptions::n_apps);
    }
  }
  else // for baseline
    for (unsigned i = 0; i < 4; pwq_app_walks[i++] = 0)
      ;
}

bool PageWalkQueue::enqueue(mem_fetch *mf)
{

  /* This might have to be changed to enforce page walker partitioning */
  PageWalker *idle_page_walker = page_walk_subsystem->get_idle_page_walker();

  /* dwsp epoch */
#if 1
  /* printf("dwsp: page walker appid = %d\n", appid); */
  /* printf("dwsp: occupancy ration  = %lf\n", get_occupancy_ratio(appid, otherappid)); */
  unsigned appid = mf->get_appID();
  unsigned otherappid = 3 - appid;
  page_walk_subsystem->occupancy_in_this_dwsp_epoch_per_app[appid] +=
    get_occupancy_ratio(appid, otherappid);
  page_walk_subsystem->number_of_enqueues_in_this_dwsp_epoch_per_app[appid]++;
#endif

  if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
    page_walk_subsystem->dwsp_epoch_tracking(mf);

  if (idle_page_walker)
  {
    idle_page_walker->initiate_page_walk(mf);
    return true;
  }
  else
  {
    memory_stats_t *mem_stat = this->mem_stat;
    unsigned app_id = mf->get_appID();
    if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      config->vm_config == VM_MASK_PLUS_DWS ||
      config->vm_config == VM_DWS_PER_APP_TLB ||
        config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
    {
      /* from page walk subsystem, get the least loaded page walker */
      /*   which has been allocated to this application. */
      /*   then try to send the page walk to this walker */
      /* DELETE THE CODE BELOW AND REPLACE WITH CODE FOR THE COMMENT */
      int walker_id = get_page_walk_queue_enqueue(mf->get_appID());
      if (walker_id == -1)
      {
        return false;
      }
      else
      {
        per_walker_queue[walker_id]->push_back(mf);
        PageWalker *pw = page_walk_subsystem->page_walkers[walker_id];
        mem_fetch *cur_mf = pw->get_current();
        if (cur_mf->get_appID() != app_id)
          mem_stat->pwq_app_intf[app_id]++;
        // return true;
      }
    }
    else //baseline architecture
    {
      assert(global_queue.size() <= size);
      if (global_queue.size() >= size)
      {
        return false;
      }
      else
      {
        global_queue.push_back(mf);
        unsigned app_id = mf->get_appID();
        pwq_app_walks[app_id]++;
        mem_stat->pwq_app_intf[app_id] += pwq_app_walks[app_id == 1 ? 2 : 1];
        // return true;
      }
    }
    // mf->set_page_walk_enqueue_time((uint64_t)gpu_sim_cycle);
    // mem_stat->pwq_tot_lat -= gpu_sim_cycle;
    // mem_stat->pwq_app_lat[app_id] -= gpu_sim_cycle;
    return true;
  }
}

mem_fetch *PageWalkQueue::dequeue(PageWalker *page_walker)
{

  mem_fetch *page_walk;
  unsigned app_id = 0;
  if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      config->vm_config == VM_MASK_PLUS_DWS ||
      config->vm_config == VM_DWS_PER_APP_TLB ||
      config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
  {
    /* dequeue from the walker queue of the page walker. the page walkers have' */
    /*   IDs which allow them to be identified */
    /* walker_id must have been ideally named walk_queue_id */
    int walker_id = get_page_walk_queue_dequeue(page_walker);
    if (walker_id == -1)
    {
      return NULL;
    }
    else
    {
      page_walk = per_walker_queue[walker_id]->front();
      per_walker_queue[walker_id]->pop_front();
      if(walker_id != page_walker->id)
      {
        page_walker->current_is_stolen = true;
      }
      else
      {
        page_walker->current_is_stolen = false;
      }
    }
  }
  else
  {
    if (global_queue.size() == 0)
    {
      return NULL;
    }
    else
    {
      page_walk = global_queue.front();
      global_queue.pop_front();
      app_id = page_walk->get_appID();
      pwq_app_walks[app_id]--;
    }
  }
  memory_stats_t *mem_stat = this->mem_stat;
  unsigned pw_queueing_lat = (gpu_sim_cycle + gpu_tot_sim_cycle) - page_walk->get_timestamp();
  mem_stat->pwq_tot_lat += pw_queueing_lat;
  mem_stat->pwq_app_lat[app_id ? app_id : page_walk->get_appID()] += pw_queueing_lat;
  return page_walk;
}

int PageWalkQueue::get_page_walk_queue_enqueue(unsigned appid)
{
  /* This is for multiple applications and multiple walkers */
  /* Find the least loaded page walker which is default allocated to
   * this application and return the page walker*/
  int light_walker = -1;
  unsigned load = size;
  for (int i = 0; i < config->concurrent_page_walks; i++)
  {
    if (appid == walker_to_app_map[i])
    {
      if (per_walker_queue[i]->size() < size &&
          per_walker_queue[i]->size() < load)
      {
        load = per_walker_queue[i]->size();
        light_walker = i;
      }
    }
  }
  return light_walker;
}

int PageWalkQueue::get_page_walk_queue_dequeue_dws(unsigned appid)
{
  /* This is for multiple applications and multiple walkers */
  /* This method selects the page walk queue from the pool of per walker
   * page walk queues and returns the page walk queue*/

  int heavy_walker = -1;
  unsigned load = 0;
  for (int i = 0; i < config->concurrent_page_walks; i++)
  {
    if (appid == walker_to_app_map[i])
    {
      if (per_walker_queue[i]->size() > 0 &&
          per_walker_queue[i]->size() > load)
      {
        load = per_walker_queue[i]->size();
        heavy_walker = i;
      }
    }
  }
  if (heavy_walker == -1)
  {
    for (int i = 0; i < config->concurrent_page_walks; i++)
    {
      if (appid != walker_to_app_map[i])
      {
        if (per_walker_queue[i]->size() > 0 &&
            per_walker_queue[i]->size() > load)
        {
          load = per_walker_queue[i]->size();
          heavy_walker = i;
        }
      }
    }
  }
  return heavy_walker;
}

double PageWalkQueue::get_occupancy_app(int appid)
{
  /* printf("appid : %d", appid); */
  unsigned load = 0, max = 0;
  for (int i = 0; i < config->concurrent_page_walks; i++)
  {
    if (appid == walker_to_app_map[i])
    {
      load += per_walker_queue[i]->size();
      max += size;
    }
  }
  return (double)load / (double)max;
}

double PageWalkQueue::get_occupancy_queue(unsigned id)
{
  unsigned load = per_walker_queue[id]->size();
  return (double)load / (double)size;
}

double PageWalkQueue::get_occupancy_ratio(int appid, int otherappid)
{
  double occupancy_app = get_occupancy_app(appid);
  double occupancy_otherapp = get_occupancy_app(otherappid);

  /* printf("occupancy app %f\n", occupancy_app); */
  /* printf("occupancy other %f\n", occupancy_otherapp ); */
  /* printf("occupancy ration %f\n", occupancy_otherapp - occupancy_app); */
  return occupancy_otherapp - occupancy_app;
}

int PageWalkQueue::get_page_walk_queue_dequeue_dws_plus(PageWalker *page_walker)
{
  /* This is for multiple applications and multiple walkers */
  /* This method selects the page walk queue from the pool of per walker
   * page walk queues and returns the page walker*/
  unsigned id = page_walker->id;
  unsigned appid = page_walker->appid;
  unsigned otherappid = 3 - appid;
  unsigned last_serviced_stolen = page_walker->last_serviced_stolen;
  int pwq = PWQ_UNASSIGNED;

  /* NEW LOGIC */

  unsigned stealing_freq = config->stealing_freq_threshold;
  double dwsp_occupancy_threshold = config->dwsp_occupancy_threshold;
  double dwsp_queue_threshold= config->dwsp_queue_threshold;

  /* double occ_diff = get_occupancy_ratio(appid, otherappid); */
  /* uint64_t occupancy_diff = int(abs((10 * occ_diff) + 0.5)); */
  /* mem_stat->occupancy_difference[occupancy_diff]++; */

  dwsp_occupancy_threshold = page_walk_subsystem->dwsp_occupancy_threshold;

  if((last_serviced_stolen < stealing_freq) && 
      (get_occupancy_queue(page_walker->id) <= dwsp_queue_threshold) &&
      (get_occupancy_ratio(appid, otherappid) > dwsp_occupancy_threshold))
  {
    /* STEAL */
    mem_stat->stealing_activated_dwsp[otherappid] ++;
    pwq = get_page_walk_queue_dequeue_dws(otherappid);
    assert(page_walker->last_serviced_stolen < stealing_freq);
    page_walker->last_serviced_stolen = page_walker->last_serviced_stolen + 1;
  }
  else
  {
    /* DONT STEAL */
    page_walker->last_serviced_stolen = 0;
    unsigned id = page_walker->id;
    if (per_walker_queue[id]->empty())
      pwq = get_page_walk_queue_dequeue_dws(page_walker->appid);
    else
      pwq = id;
  }

  /* OLD LOGIC */
  /* has been removed for aesthetic purposes */

  assert(pwq != PWQ_UNASSIGNED && "should have a valid pwq to dequeue from\n");
  if (walker_to_app_map[pwq] != appid)
    mem_stat->pwq_app_intf[appid] += per_walker_queue[id]->size();
  return pwq;
}

int PageWalkQueue::get_page_walk_queue_dequeue(PageWalker *page_walker)
{
  int pwq = PWQ_UNASSIGNED;
  unsigned id = page_walker->id;
  if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      config->vm_config == VM_MASK_PLUS_DWS ||
      config->vm_config == VM_DWS_PER_APP_TLB )
  {
    if (per_walker_queue[id]->empty())
      pwq = get_page_walk_queue_dequeue_dws(page_walker->appid);
    else
      pwq = id;
  }
  else if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
  {
    pwq = get_page_walk_queue_dequeue_dws_plus(page_walker);
  }

  if (pwq != PWQ_RESERVATION_FAIL && pwq != id) //if the queue returned does not belong to the walker
  {
    unsigned cur_app = walker_to_app_map[id];
    memory_stats_t *mem_stat = this->mem_stat;
    int pwq_owner = walker_to_app_map[pwq];
    if (pwq_owner == cur_app)
      mem_stat->pwq_app_pw_stolen[PW_STOLEN_FROM_SELF][cur_app]++;
    else
    {
      mem_stat->pwq_app_pw_stolen[PW_STOLEN_FROM_OTHER][cur_app]++;
      page_walk_subsystem->number_of_walks_serviced_by_stealing[pwq_owner]++;
    }
    mem_stat->pwq_app_pw_stolen[PW_TOT_STOLEN][cur_app]++;
  }
  assert(pwq != PWQ_UNASSIGNED && "should have a valid pwq to dequeue from\n");
  return pwq;
}

void PageWalkQueue::print()
{
  printf("Page Walk Queue\n");
  if (config->vm_config == VM_DYNAMIC_WALKER_SYSTEM ||
      config->vm_config == VM_MASK_PLUS_DWS ||
      config->vm_config == VM_DWS_PER_APP_TLB ||
      config->vm_config == VM_DYNAMIC_WALKER_SYSTEM_PLUS)
  {
    for (int i = 0; i < config->concurrent_page_walks; i++)
    {
      printf("Page Walk Queue : %d\n", i);
      for (std::list<mem_fetch *>::iterator it = per_walker_queue[i]->begin();
           it != per_walker_queue[i]->end(); it++)
      {
        printf("Entry: %p\n", (*it));
      }
      printf("\n");
    }
  }
  else
  {
    for (std::list<mem_fetch *>::iterator it = global_queue.begin();
         it != global_queue.end(); it++)
    {
      printf("Entry: %p\n", (*it));
    }
    printf("\n");
  }
}

void PageWalkSubsystem::flush(unsigned appid)
{
  page_walk_cache->flush(appid);
}

unsigned get_app_from_key(new_addr_type key)
{
  unsigned app = key & 15;
  assert(app == 1 || app == 2);
  return app;
}

void PageWalkCacheImpl::flush(unsigned appid)
{
  for (unsigned i = 0; i < pw_cache_entries; i++)
  {
    for (auto j = pw_cache[i]->begin(); j != pw_cache[i]->end();)
    {
      if ((get_app_from_key((*j))) == appid)
      {
        j = pw_cache[i]->erase(j);
      }
      else
        j++;
    }
  }
  // state_tracker[appid] = 0;
}
