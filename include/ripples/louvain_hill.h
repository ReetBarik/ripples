//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
//
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_LOUVAIN_HILL_H
#define RIPPLES_LOUVAIN_HILL_H

#include <queue>
#include <string>
#include <type_traits>
#include <vector>

#include "ripples/hill_climbing_engine.h"
#include "ripples/hill_climbing.h"
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

namespace ripples {

struct LouvainHillConfiguration : public HillClimbingConfiguration {
  std::string communityList;
  void addCmdOptions(CLI::App &app) {
    AlgorithmConfiguration::addCmdOptions(app);

    app.add_option("--community-map", communityList,
                   "The filename of the community map.")
        ->required()
        ->group("Algorithm Options");
  }
};

// struct LouvainIMMExecutionRecord : public IMMExecutionRecord {};

namespace {
template <typename vertex_type>
struct Compare {
  bool operator()(std::pair<vertex_type, size_t> &a,
                  std::pair<vertex_type, size_t> &b) const {
    return a.second < b.second;
  }
};
}  // namespace


template <typename GraphTy, typename RecordTy, typename ConfTy, typename execution_tag>
std::vector<typename GraphTy::vertex_type> FindMostInfluentialSeedSet(const std::vector<GraphTy> &communities, std::vector<std::vector<Bitmask<int>>> &sampled_graphs,
                            std::vector<RecordTy> &R, ConfTy &CFG,
                            execution_tag &&ex_tag) {
  spdlog::get("console")->info("SeedSelect start");

  using vertex_type = typename GraphTy::vertex_type;

  Compare<vertex_type> cmp;


  spdlog::get("console")->flush();

  // Init on heap per community
  using vertex_contribution_pair = std::pair<vertex_type, size_t>;
  std::vector<vertex_contribution_pair> global_heap(
      CFG.k + 1, vertex_contribution_pair{-1, -1});
  std::vector<uint64_t> active_communities(communities.size(), 1);

  auto heap_cmp = [](const vertex_contribution_pair &a,
                     const vertex_contribution_pair &b) -> bool {
    return a.second > b.second;
  };

  std::make_heap(global_heap.begin(), global_heap.end(), heap_cmp);
  std::mutex global_heap_mutex;

  using GraphFwd =
      ripples::Graph<uint32_t, ripples::WeightedDestination<uint32_t, float>, ripples::ForwardDirection<uint32_t>>;

  std::vector<SeedSelectionEngine<GraphFwd, std::vector<Bitmask<int>>::iterator>> SEV;
  SEV.reserve(communities.size());
  for (size_t i = 0; i < communities.size(); ++i) {
    SeedSelectionEngine<GraphFwd, std::vector<Bitmask<int>>::iterator> S(communities[i], CFG.streaming_workers, CFG.streaming_gpu_workers, "SeedSelectionEngine" + std::to_string(i));
    SEV.push_back(move(S));
  }
  while (!std::all_of(active_communities.begin(), active_communities.end(), [](const uint64_t &v) -> bool { return v == 0; })) {

// #pragma omp parallel for schedule(dynamic)
    

    for (size_t i = 0; i < communities.size(); ++i) {
      if (active_communities[i] == 0) continue;

     
        
        // vertex_contribution_pair vcp = SeedSelection(communities[i], sampled_graphs[i].begin(), sampled_graphs[i].end(), CFG,
        //                  R[i], community_seeds[i]);

        vertex_contribution_pair vcp = SEV[i].get_next_seed(sampled_graphs[i].begin(), sampled_graphs[i].end(), R[i].SeedSelectionTasks);
      


      // Handle the global index insertion
      std::lock_guard<std::mutex> _(global_heap_mutex);
      std::pop_heap(global_heap.begin(), global_heap.end(), heap_cmp);
      global_heap.back() = vcp;
      std::push_heap(global_heap.begin(), global_heap.end(), heap_cmp);

      if (global_heap.front() == vcp) active_communities[i] = 0;
    }
  }

  std::pop_heap(global_heap.begin(), global_heap.end(), heap_cmp);
  global_heap.pop_back();

  double coverage = 0;
  std::vector<typename GraphTy::vertex_type> seeds;
  seeds.reserve(CFG.k);
  std::sort_heap(global_heap.begin(), global_heap.end(), heap_cmp);
  for (auto e : global_heap) {  
    seeds.push_back(e.first);
    coverage += e.second;
  }


  return seeds;
}

// template <typename GraphTy, typename ConfTy, typename GeneratorTy,
//           typename RecordTy, typename diff_model_tag>
// auto LouvainHill(const std::vector<GraphTy> &communities, ConfTy &CFG, 
//                 GeneratorTy &gen, diff_model_tag &&model_tag,
//                 sequential_tag &&ex_tag) {
//   //TODO:: Implement
// }
//! Influence Maximization using Community Structure.
//!
//! The algorithm uses the Louvain method for community detection and then
//! IMM to select seeds frome the communities.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param communities The input graphs.  The graphs are transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename RecordTy,
           typename GeneratorTy, typename diff_model_tag>
auto LouvainHill(const std::vector<GraphTy> &communities, ConfTy &CFG, std::vector<RecordTy> &R,
                GeneratorTy gen, diff_model_tag &&model_tag,
                omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;

  std::vector<decltype(gen)> comm_gen(communities.size());

  for (size_t i = 0; i < communities.size(); ++i) {
    auto local_gen = gen;
    local_gen.split(communities.size(), i);
    comm_gen[i] = local_gen;
  }

  std::vector<std::vector<Bitmask<int>>> sampled_graphs(communities.size()); 
  
  // For each community do Sampling
// #pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < communities.size(); ++i) {
    sampled_graphs[i] = SampleFrom(communities[i], CFG, comm_gen[i], R[i], std::forward<diff_model_tag>(model_tag), i);
  }

  // Global seed selection using the heap
  auto S = FindMostInfluentialSeedSet(communities, sampled_graphs, R, CFG,
                                  std::forward<omp_parallel_tag>(ex_tag));

  return std::make_pair(S, R);
}

}  // namespace ripples

#endif /* RIPPLES_LOUVAIN_HILL_H */
