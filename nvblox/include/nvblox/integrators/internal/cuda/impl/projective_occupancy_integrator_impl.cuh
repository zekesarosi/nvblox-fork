/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <nvblox/integrators/projective_occupancy_integrator.h>

#include <limits>

#include <nvblox/integrators/internal/cuda/impl/projective_integrator_impl.cuh>

#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/occupancy_integrator_params.h"

namespace nvblox {

struct UpdateOccupancyVoxelFunctor {
  UpdateOccupancyVoxelFunctor() {}

  __device__ bool operator()(const float surface_depth_measured,
                             const float voxel_depth_m, const bool is_active,
                             OccupancyVoxel* voxel_ptr) {
    if (surface_depth_measured <= 0.F) {
      return false;
    }

    // Get the update summand depending on the measured depth
    float log_odds_update;

    // Organized no-return beams write a far sentinel depth so the ray still
    // carves. At 1 m voxels a thin pole shares a cell with sky, and the normal
    // free prior lets those miss rays erase the pole, so misses get their own
    // weaker free vote: hits still win while the voxel is observed, and sky
    // desticks slowly once the vehicle has passed.
    //
    // The threshold sits just under the sentinel rather than at some fixed
    // range, so a real long-range return keeps the full free prior and the
    // classification cannot silently change when the sentinel is retuned.
    // miss_ray_min_depth_m_ is +inf when miss-ray carving is disabled.
    if (surface_depth_measured >= miss_ray_min_depth_m_) {
      log_odds_update = miss_ray_log_odds_;
    } else if (!is_active || voxel_depth_m > surface_depth_measured +
                                          occupied_region_half_width_m_) {
      log_odds_update = unobserved_region_log_odds_;
    } else if (voxel_depth_m >
               surface_depth_measured - occupied_region_half_width_m_) {
      log_odds_update = occupied_region_log_odds_;
    } else {
      log_odds_update = free_region_log_odds_;
    }

    // Update and clip
    float updated_log_odds = voxel_ptr->log_odds + log_odds_update;
    voxel_ptr->log_odds =
        fmax(kMinLogOdds_, fmin(updated_log_odds, kMaxLogOdds_));

    return true;
  }

  // Sensor model parameters
  float free_region_log_odds_ = logOddsFromProbability(
      kFreeRegionOccupancyProbabilityParamDesc.default_value);
  float occupied_region_log_odds_ = logOddsFromProbability(
      kOccupiedRegionOccupancyProbabilityParamDesc.default_value);
  float unobserved_region_log_odds_ = logOddsFromProbability(
      kUnobservedRegionOccupancyProbabilityParamDesc.default_value);
  float occupied_region_half_width_m_ =
      kOccupiedRegionHalfWidthMParamDesc.default_value;
  float miss_ray_log_odds_ = logOddsFromProbability(
      kMissRayOccupancyProbabilityParamDesc.default_value);
  float miss_ray_min_depth_m_ = std::numeric_limits<float>::infinity();

  // Min and max values for clipping
  const float kMaxLogOdds_ = logOddsFromProbability(0.99);
  const float kMinLogOdds_ = logOddsFromProbability(0.01);
};

template <typename SensorType>
void ProjectiveOccupancyIntegrator::integrateFrame(
    const MaskedDepthImageConstView& depth_frame, const Transform& T_L_C,
    const SensorType& sensor, OccupancyLayer* layer,
    std::vector<Index3D>* updated_blocks) {
  setFunctorParameters(layer->voxel_size());
  ProjectiveIntegrator<OccupancyVoxel>::integrateFrame(
      depth_frame, T_L_C, sensor,
      update_functor_host_ptr_.cloneAsync(MemoryType::kDevice, *cuda_stream_)
          .get(),
      layer, updated_blocks);
}

}  // namespace nvblox
