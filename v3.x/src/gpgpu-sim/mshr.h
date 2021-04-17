
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

#ifndef MSHR_H
#define MSHR_H

#include "gpu-misc.h"
#include "mem_fetch.h"
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"
#include <stdio.h>
#include <stdlib.h>

class mshr_table {
public:
  mshr_table():
      m_num_entries(12), m_max_merged(4)
#if (tr1_hash_map_ismap == 0)
          , m_data(2 * 12)
#endif
  {
  }
  mshr_table(unsigned num_entries, unsigned max_merged) :
      m_num_entries(num_entries), m_max_merged(max_merged)
#if (tr1_hash_map_ismap == 0)
          , m_data(2 * num_entries)
#endif
  {
  }

  /// Checks if there is a pending request to the lower memory level already
  bool probe(new_addr_type block_addr) const;
  /// Checks if there is space for tracking a new memory access
  bool full(new_addr_type block_addr) const;
  /// Add or merge this access
  void add(new_addr_type block_addr, mem_fetch *mf);
  /// Returns true if cannot accept new fill responses
  bool busy() const {
    return false;
  }
  /// Accept a new cache fill response: mark entry ready for processing
  void mark_ready(new_addr_type block_addr, bool &has_atomic);
  /// Returns true if ready accesses exist
  bool access_ready() const {
    return !m_current_response.empty();
  }
  /// Returns next ready access
  mem_fetch *next_access();
  void display(FILE *fp) const;

  void check_mshr_parameters(unsigned num_entries, unsigned max_merged) {
    assert(
        m_num_entries == num_entries && "Change of MSHR parameters between kernels is not allowed");
    assert(
        m_max_merged == max_merged && "Change of MSHR parameters between kernels is not allowed");
  }

  //pratheek
  void mark_ready(new_addr_type key);

private:

  // finite sized, fully associative table, with a finite maximum number of merged requests
  const unsigned m_num_entries;
  const unsigned m_max_merged;

  struct mshr_entry {
    std::list<mem_fetch*> m_list;
    bool m_has_atomic;
    mshr_entry() :
        m_has_atomic(false) {
    }
  };
  typedef tr1_hash_map<new_addr_type,mshr_entry> table;
  table m_data;

  // it may take several cycles to process the merged requests
  bool m_current_response_ready;
  std::list<new_addr_type> m_current_response;
};

#endif
