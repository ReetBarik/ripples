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
#include "spdlog/async.h"
#include "mpi.h"
#ifdef RIPPLES_ENABLE_CUDA
#include "cuda_runtime.h"
#include "ripples/cuda/cuda_utils.h"
#endif

namespace ripples {

struct LouvainHillConfiguration : public HillClimbingConfiguration {
  std::string communityList;
  size_t num_threads_d1{4};
  void addCmdOptions(CLI::App &app) {
    AlgorithmConfiguration::addCmdOptions(app);

    app.add_option("--community-map", communityList,
                   "The filename of the community map.")
        ->required()
        ->group("Algorithm Options");

    app.add_option(
       "--streaming-gpu-workers", streaming_gpu_workers,
       "The number of GPU workers for the CPU+GPU streaming engine.")
    ->group("Streaming-Engine Options");

    app.add_option(
       "--concurrent-partitions", num_threads_d1,
       "The number of partitions to be processed parallely.")
    ->group("Streaming-Engine Options");
  }
};

template <typename vertex_type>
std::pair<vertex_type, size_t>* merge (std::pair<vertex_type, size_t>* a1, std::pair<vertex_type, size_t>* a2, int size) {

	std::pair<vertex_type, size_t>* temp = (std::pair<vertex_type, size_t>*)malloc(2 * sizeof(std::pair<vertex_type, size_t>) * size);
	int i = 0, j = 0, h = 0;

	while(i < size && j < size) {

		if(a1[i].second < a2[j].second) {
			temp[h].first = a1[i].first;
			temp[h].second = a1[i].second;
			i++; h++;
		}
		else {
			temp[h].first = a2[j].first;
			temp[h].second = a2[j].second;
			j++; h++;
		}
	}

	while (i < size) {
		temp[h].first = a1[i].first;
		temp[h].second = a1[i].second;
		i++; h++;
	}

	while (j < size) {
		temp[h].first = a2[j].first;
		temp[h].second = a2[j].second;
		j++; h++;
	}

	return temp;

}

template <typename vertex_type>
std::pair<vertex_type, size_t>* MyAllReduce(int rank, int size, std::pair<vertex_type, size_t> *localArray, int p) {

	int partner;
	std::pair<vertex_type, size_t>* result = (std::pair<vertex_type, size_t>*)malloc(sizeof(std::pair<vertex_type, size_t>) * size);
	std::pair<vertex_type, size_t>* temp;
	MPI_Status status;

	
	for (int t = 0; t < ceil(log2(p)); t++) {																																						// iterate through log2(p) steps

		partner = rank ^ (1 << t);																																									// XOR and bitshift

		MPI_Sendrecv(localArray, sizeof(std::pair<vertex_type, size_t>) * size, MPI_BYTE, partner, 0, result, sizeof(std::pair<vertex_type, size_t>) * size, MPI_BYTE, partner, 0, MPI_COMM_WORLD, &status);    	// exchange with partner process
		
		temp = merge(localArray, result, size);

		for (int i = size; i < 2 * size; i++) {
			localArray[i - size].first = temp[i].first;
			localArray[i - size].second = temp[i].second;
		}

	}

	for (int i = 0; i < size; i++) {
		result[i].first = localArray[i].first;
		result[i].second = localArray[i].second;
	}
	
	return result;		

}

namespace {
template <typename vertex_type>
struct Compare {
  bool operator()(std::pair<vertex_type, size_t> &a,
                  std::pair<vertex_type, size_t> &b) const {
    return a.second > b.second;
  }
};
}  // namespace


template <typename GraphTy, typename RecordTy, typename ConfTy, typename execution_tag>
std::vector<typename GraphTy::vertex_type> FindMostInfluentialSeedSet(std::vector<GraphTy> & communities, std::vector<std::vector<Bitmask<int>>> &sampled_graphs,
                            std::vector<RecordTy> &R, ConfTy &CFG,
                            execution_tag &&ex_tag) {
  spdlog::get("console")->info("SeedSelect start");

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  using vertex_type = typename GraphTy::vertex_type;


  Compare<vertex_type> cmp;
  
  omp_set_nested(1);
  int num_threads_d1 = CFG.num_threads_d1, num_threads_d2;
  
  num_threads_d2 = std::floor(omp_get_max_threads() / num_threads_d1);

  
  size_t total_gpu = 0; 
  #if RIPPLES_ENABLE_CUDA
  total_gpu = int(cuda_num_devices() / num_threads_d1) * num_threads_d1;
  CFG.streaming_gpu_workers = total_gpu / num_threads_d1;
  #endif
  spdlog::get("console")->flush();

  CFG.streaming_workers = num_threads_d2;
  CFG.streaming_workers -= CFG.streaming_gpu_workers;

  // Init on heap per community
  using vertex_contribution_pair = std::pair<vertex_type, size_t>;
  std::pair<vertex_type, size_t>* global_heap = (std::pair<vertex_type, size_t>*)malloc(sizeof(std::pair<vertex_type, size_t>) * (CFG.k + 1));
      
  std::vector<vertex_contribution_pair> local_heap(
      CFG.k + 1, vertex_contribution_pair{-1, 0});
  std::vector<uint64_t> active_communities(communities.size(), 1);

  auto heap_cmp = [](const vertex_contribution_pair &a,
                     const vertex_contribution_pair &b) -> bool {
    return a.second > b.second;
  };

  std::make_heap(local_heap.begin(), local_heap.end(), heap_cmp);
  std::mutex local_heap_mutex;
  
  using GraphFwd =
      ripples::Graph<uint32_t, ripples::WeightedDestination<uint32_t, float>, ripples::ForwardDirection<uint32_t>>;

  std::vector<SeedSelectionEngine<GraphFwd, std::vector<Bitmask<int>>::iterator>*> SEV;
  SEV.reserve(communities.size());
  for (size_t i = 0; i < communities.size(); ++i) {
    auto S = new SeedSelectionEngine<GraphFwd, std::vector<Bitmask<int>>::iterator>(communities[i], CFG.streaming_workers, CFG.streaming_gpu_workers, "SeedSelectionEngine" + std::to_string(i), CFG.streaming_gpu_workers, (i % num_threads_d1) * CFG.streaming_gpu_workers);
    SEV[i] = S;
  }

  bool firstIter = false;

  if (communities.size() < CFG.k + 1) firstIter = true;

  while (!std::all_of(active_communities.begin(), active_communities.end(), [](const uint64_t &v) -> bool { return v == 0; })) {

  

  	for (size_t j = 0; j < (firstIter ? std::ceil((CFG.k + 1) / communities.size()) : 1) ; j++) {

#pragma omp parallel for schedule(static) num_threads(num_threads_d1) 
	    for (size_t i = 0; i < communities.size(); ++i) {
	      if (active_communities[i] == 0) continue;

	      vertex_contribution_pair vcp = SEV[i]->get_next_seed(sampled_graphs[i].begin(), sampled_graphs[i].end(), R[i].SeedSelectionTasks);
	      if (vcp.first == -1) {
	        active_communities[i] = 0;
	        continue;
	      }
	      vcp.first = communities[i].convertID(vcp.first);


	      // Handle the global index insertion
	      std::lock_guard<std::mutex> _(local_heap_mutex);
	      std::pop_heap(local_heap.begin(), local_heap.end(), heap_cmp);

	      local_heap.back() = vcp;
	      std::push_heap(local_heap.begin(), local_heap.end(), heap_cmp);

	      if (local_heap.front() == vcp) active_communities[i] = 0;
	    }
	}

    std::sort_heap(local_heap.begin(), local_heap.end(), heap_cmp);
    // ALL REDUCE 
    vertex_contribution_pair* h = &local_heap[0];
    global_heap = MyAllReduce(world_rank, CFG.k + 1, h, world_size);
    
    for (size_t j = 0; j < CFG.k + 1; j++) {
    	local_heap[j].first = global_heap[j].first;
    	local_heap[j].second = global_heap[j].second;
    }
    std::make_heap(local_heap.begin(), local_heap.end(), heap_cmp);

  }

  std::pop_heap(local_heap.begin(), local_heap.end(), heap_cmp);

  local_heap.pop_back();

  double coverage = 0;
  std::vector<typename GraphTy::vertex_type> seeds;
  seeds.reserve(CFG.k);
  std::sort_heap(local_heap.begin(), local_heap.end(), heap_cmp);
  for (auto e : local_heap) {  
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
auto LouvainHill(std::vector<GraphTy> &communities, ConfTy &CFG, std::vector<RecordTy> &R,
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
