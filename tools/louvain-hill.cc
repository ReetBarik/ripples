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

#include <algorithm>
#include <fstream>
#include <iterator>

#include "ripples/configuration.h"
#include "ripples/diffusion_simulation.h"
#include "ripples/graph.h"
#include "ripples/loaders.h"
#include "ripples/louvain_hill.h"
#include "ripples/utility.h"

#include "omp.h"

namespace ripples {


template <typename SeedSet>
auto GetExperimentRecord(
    const ToolConfiguration<HillClimbingConfiguration>& CFG,
    const HillClimbingExecutionRecord& R, const SeedSet& seeds) {
  nlohmann::json experiment{{"Algorithm", "HillClimbing"},
                            {"Input", CFG.IFileName},
                            {"Output", CFG.OutputFile},
                            {"DiffusionModel", CFG.diffusionModel},
                            {"K", CFG.k},
                            {"Seeds", seeds},
                            {"NumThreads", R.NumThreads},
                            {"NumWalkWorkers", CFG.streaming_workers},
                            {"NumGPUWalkWorkers", CFG.streaming_gpu_workers},
                            {"Total", R.Total},
                            {"Sampling", R.Sampling},
                            {"SeedSelection", R.SeedSelection},
                            {"SamplingTasks", R.SamplingTasks},
                            {"SeedSelectionTasks", R.SeedSelectionTasks}};

  return experiment;
}

}  // namespace ripples

ripples::ToolConfiguration<ripples::LouvainHillConfiguration> CFG;

void parse_command_line(int argc, char **argv) {
  CFG.ParseCmdOptions(argc, argv);
#pragma omp single
  CFG.streaming_workers = omp_get_max_threads();
  // CFG.streaming_gpu_workers = CFG.streaming_workers;
  // if (CFG.seed_select_max_workers == 0)
  //   CFG.seed_select_max_workers = CFG.streaming_workers;
  // if (CFG.seed_select_max_gpu_workers == std::numeric_limits<size_t>::max())
  //   CFG.seed_select_max_gpu_workers = CFG.streaming_gpu_workers;
}

int main(int argc, char *argv[]) {
  auto console = spdlog::stdout_color_st("console");
  parse_command_line(argc, argv);

  spdlog::set_level(spdlog::level::info);

  trng::lcg64 weightGen;
  weightGen.seed(0UL);
  weightGen.split(2, 0);

  using dest_type = ripples::WeightedDestination<uint32_t, float>;
  using GraphFwd =
      ripples::Graph<uint32_t, dest_type, ripples::ForwardDirection<uint32_t>>;
  
  console->info("Loading...");
  GraphFwd Gf = ripples::loadGraph<GraphFwd>(CFG, weightGen);
  console->info("Loading Done!");
  console->info("Number of Nodes : {}", Gf.num_nodes());
  console->info("Number of Edges : {}", Gf.num_edges());

  std::vector<typename GraphFwd::vertex_type> communityVector;
  communityVector.reserve(Gf.num_nodes());

  {
    std::ifstream f(CFG.communityList);
    std::istream_iterator<typename GraphFwd::vertex_type> eos;
    std::istream_iterator<typename GraphFwd::vertex_type> itr(f);

    std::copy(itr, eos, std::back_inserter(communityVector));
  }

  console->info("Communities Vector Size : {}", communityVector.size());

  const auto communities =
      ripples::getCommunitiesSubgraphs<GraphFwd>(Gf, communityVector);
  console->info("Number of Communities : {}", communities.size());
  omp_get_max_threads() > communities.size() ? omp_set_num_threads(communities.size()) : omp_set_num_threads(omp_get_max_threads());
  nlohmann::json executionLog;

  std::vector<typename GraphFwd::vertex_type> seeds;
  std::vector<ripples::HillClimbingExecutionRecord> R(communities.size());

  trng::lcg64 generator;
  generator.seed(0UL);
  generator.split(2, 1);
  
  std::ofstream perf(CFG.OutputFile);
  if (CFG.parallel) {
    auto workers = (communities.size() < CFG.streaming_workers ? communities.size() : CFG.streaming_workers);
    // CFG.seed_select_max_workers = workers;
    auto gpu_workers = CFG.streaming_gpu_workers;

    if (CFG.diffusionModel == "IC") {
      auto start = std::chrono::high_resolution_clock::now();


      
      std::tie(seeds, R) = LouvainHill(communities, CFG, R, generator, 
                                      ripples::independent_cascade_tag{},
                                      ripples::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R[0].Total = end - start;
    } else if (CFG.diffusionModel == "LT") {
      auto start = std::chrono::high_resolution_clock::now();
      
      std::tie(seeds, R) =
          LouvainHill(communities, CFG, R, generator, ripples::linear_threshold_tag{},
                     ripples::omp_parallel_tag{});
      auto end = std::chrono::high_resolution_clock::now();
      R[0].Total = end - start;
    }
    console->info("Louvain IMM parallel : {}ms", R[0].Total.count());

  } else {
    //TODO:: Not done yet
  }

  return EXIT_SUCCESS;
}
