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

#ifndef GPU_PAGEWALK_H
#define GPU_PAGEWALK_H

#include "gpu-sim.h"

class mem_fetch;

class tlb_tag_array;

struct memory_config;

class PageWalker;
class PageWalkCache;
class PageWalkQueue;

/* Only PageWalkSubsystem and PageWalker classes must be exposed to the outer
 * world. Specifically, the PageWalkSubsystem class is exposed to the L2 TLB
 * and the page_walk_return() method of PageWalker class is exposed to
 * mem_fetch class. The latter is necessary to simplify code by NOT involving
 * the PageWalkSubsystem on page_walk_return.
 * */

class PageWalkSubsystem
{

  /*   We to create as many page walkers.
   * One page walk cache object which internally could be two or one page walk
   * caches.
   * One page walk queue object which internally could implement multiple or
   * single queue.
   * */

  friend class PageWalker;
  friend class PageWalkCache;
  friend class PageWalkQueue;

private:
  std::vector<PageWalker *> page_walkers;
  std::list<PageWalker*> stealing_latency_queue;
  PageWalkCache *page_walk_cache;
  PageWalkQueue *page_walk_queue;
  struct tlb_tag_array *L2_TLB;
  mmu *page_manager;
  const memory_config *config;
  memory_stats_t *mem_stat;
  unsigned num_walkers;
  /* This (current_allocation) is used to track the number of page walkers
     * assigned to an application currently.
     * */
public:
  unsigned current_allocation[4];

  /*   The enqueue interface is used by the L2 TLB to send page walk requests
   * downsteams to be enqueued, or directly sent to the selected pagewalker.
   * The dequeue interface is used by the PageWalker class to get the next
   * page walk to service.
   * */

public:
  PageWalkSubsystem(tlb_tag_array *tlb, mmu *page_manager,
                    const memory_config *config, memory_stats_t *mem_stat);
  bool enqueue(mem_fetch *mf);
  mem_fetch *dequeue(PageWalker *page_walker);
  unsigned get_num_walkers();
  struct tlb_tag_array *get_tlb();
  PageWalker *get_idle_page_walker();
  void service_page_walk_cache_queue();
  void page_walk_cache_enqueue(mem_fetch *mf);
  void flush(unsigned appid);
  /* We require an interface to get_page_walk_queue_enqueue from the page
     * walk subsystem interface so that it can be accessed from the TLB */
  unsigned get_page_walk_queue_enqueue(mem_fetch *mf);
  void stealing_latency_pause(PageWalker* page_walker);
  void stealing_latency_resume();
  void dwsp_epoch_tracking(mem_fetch* mf);

/* TODO:intialilize */
  unsigned page_walk_in_this_dwsp_epoch;
  unsigned page_walk_in_this_dwsp_epoch_per_app[4];
  unsigned dwsp_epoch_number;
  double dwsp_occupancy_threshold;

  double occupancy_in_this_dwsp_epoch_per_app[4];
  unsigned number_of_enqueues_in_this_dwsp_epoch_per_app[4];
  unsigned number_of_walks_serviced_by_stealing[4];
};

class PageWalker
{

  /*   Stores the state of page walk being serviced.
   * The method get_current() returns the mem_fetch that is currently being
   * serviced by the page walker.
   * When a page walk completes, the method page_walk_return is used for
   * multiple actions.
   * -> Fill into the L1 and L2 TLBs.
   * -> Get the next page to be serviced from the page walk queue.
   * On a side note, appid is mostly useless.
   *  */

  friend class PageWalkSubsystem;

private:
  mem_fetch *current;
  const memory_config *config;
  memory_stats_t *mem_stat;
  PageWalkSubsystem *page_walk_subsystem;
  PageWalkCache *page_walk_cache;

public:
  unsigned id;
  unsigned appid;
  /* number of times serviced stolen page walks */
  unsigned last_serviced_stolen;
  bool current_is_stolen;
  unsigned long long previous_translation_finish_cycle;
  PageWalker(PageWalkSubsystem *page_walk_subsystem,
             const memory_config *config, memory_stats_t *mem_stat, unsigned id);
  mem_fetch *get_current();
  bool page_walk_return(mem_fetch *mf);
  void initiate_page_walk(mem_fetch *mf);
  mem_fetch *pt_walk(mem_fetch *mf);
  void probe_pw_cache(mem_fetch *mf);
  /* mem_fetch* done_tlb_req(mem_fetch* mf); */
  void print();
};

class PageWalkCacheImpl
{

  /*   This is the actual implementation of the page walk cache arrays.
   * Each page walk cache array is accompanied by one PWC latency queue and
   * another PWC latency time queue. These are required for emulating the
   * latency of the page walk cache hits.
   * The methods are mostly the implementation of the encompassing class. 
   *  */
private:
  std::list<new_addr_type> **pw_cache;
  std::list<mem_fetch *> *pw_cache_lat_queue;
  std::list<unsigned long long> *pw_cache_lat_time;
  const memory_config *config;
  memory_stats_t *mem_stat;
  unsigned ports;
  unsigned pw_cache_entries;
  unsigned pw_cache_ways;

public:
  PageWalkCacheImpl(const memory_config *config, memory_stats_t *mem_stat);
  bool access(new_addr_type key, unsigned index);
  bool fill(new_addr_type key, unsigned index);
  void service_latency_queue();
  void enqueue(mem_fetch *mf);
  void print();
  void flush(unsigned appid);
};

class PageWalkCache
{

  /*   We have three possible page walk cache configurations.
   * Disabled.
   * Page walk cache is shared between all applications and walkers.
   * Each application is allocated its own page walk cache
   * 
   * The parameter `ports` determines how many times the page walk cache can be
   * pumped per cycle.
   * The ports are per internal page walk cache; ie., if there are two internal
   * page walk caches, then each of them can be pumped `ports` number of times
   * every clock.
   * */

private:
  const memory_config *config;
  memory_stats_t *mem_stat;
  unsigned ports;
  unsigned pw_cache_entries;
  unsigned pw_cache_ways;
  bool per_app_pwc_used;
  PageWalkSubsystem *page_walk_subsystem;
  mmu *page_manager;
  PageWalkCacheImpl *pwc;
  PageWalkCacheImpl **per_app_pwc;
  unsigned state_tracker[4];

  /*   The access() method accesses a page walk cache entry and promotes it to
   * the MRU position.
   * The fill() method is used to fill an entry into the page walk cache.
   * To index into the page walk cache, the get_key() method returns the index
   * based on the address in the nth level page walk mem_fetch.
   * The PageWakler::probe_pw_cache() method is used to check if a particular 
   * mid level page walk * hits in the page walk cache.
   * The methods accesses() and * fills() are used by the the probe_pw_cache()
   * to lookup and fill into the page walk cache from the PageWalker.
   * Finally, the service_latency_queue() is called every cycle to emulate the
   * latency of page walk cache hits and service the page walks which hit in
   * the page walk cache.
   * */

public:
  PageWalkCache(PageWalkSubsystem *page_walk_subsystem,
                const memory_config *config, memory_stats_t *mem_stat);
  bool access(mem_fetch *mf);
  bool fill(mem_fetch *mf);
  new_addr_type get_key(mem_fetch *mf);
  void service_latency_queue();
  void enqueue(mem_fetch *mf);
  void print();
  void flush(unsigned appid) { pwc->flush(appid); }

};

class PageWalkQueue
{

  /*   This class is the container class for the queue(s) where page walk
   * misses from the L2 TLB are stored before they are being serviced. 
   * The actual implementation of queues and the policies regaring their
   * servicing are abstractd away from the TLB and page walkers. 
   * Based on the configuration, we can have either a global queue of page walk
   * requests or per page walker queue.
   * The enqueue() function interfaces with the L2 TLB misses (through the
   * page walk subsystem class enqueue().
   * The dequeue () is used by the returning page walkers to get their next
   * page walk to service.
   *  */

private:
  bool per_walker_queue_used;
  std::list<mem_fetch *> global_queue;
  uint64_t pwq_app_walks[4];
  std::vector<std::list<mem_fetch *> *> per_walker_queue;
  /* We are assuming the max number of walkers to be 32 */
  unsigned walker_to_app_map[32];
  unsigned size;
  const memory_config *config;
  memory_stats_t *mem_stat;
  PageWalkSubsystem *page_walk_subsystem;
  int get_page_walk_queue_dequeue_dws(unsigned appid);
  int get_page_walk_queue_dequeue_dws_plus(PageWalker *page_walker);
  double get_occupancy_app(int appid);
  double get_occupancy_ratio(int appid, int otherappid);
  double get_occupancy_queue(unsigned id);
  enum
  {
    PW_STOLEN_FROM_SELF,
    PW_STOLEN_FROM_OTHER,
    PW_TOT_STOLEN
  };
  enum
  {
    PWQ_UNASSIGNED = -2,
    PWQ_RESERVATION_FAIL = -1
  };

public:
  int get_page_walk_queue_enqueue(unsigned appid);
  int get_page_walk_queue_dequeue(PageWalker *page_walker);

public:
  PageWalkQueue(PageWalkSubsystem *page_walk_subsystem,
                const memory_config *config, memory_stats_t *mem_stat);
  bool enqueue(mem_fetch *mf);
  mem_fetch *dequeue(PageWalker *page_walker);
  void print();
};

#endif
