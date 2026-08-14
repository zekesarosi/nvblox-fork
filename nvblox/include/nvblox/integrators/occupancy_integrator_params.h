/*
Copyright 2024 NVIDIA CORPORATION

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

#include "nvblox/utils/params.h"

namespace nvblox {
constexpr Param<float>::Description kFreeRegionOccupancyProbabilityParamDesc{
    "free_region_occupancy_probability", 0.3f,
    "The inverse sensor model occupancy probability for voxels observed as "
    "free space."};

constexpr Param<float>::Description
    kOccupiedRegionOccupancyProbabilityParamDesc{
        "occupied_region_occupancy_probability", 0.7f,
        "The inverse sensor model occupancy probability for voxels observed as "
        "occupied."};

constexpr Param<
    float>::Description kUnobservedRegionOccupancyProbabilityParamDesc{
    "unobserved_region_occupancy_probability", 0.5f,
    "The inverse sensor model occupancy probability for unobserved voxels."};

constexpr Param<float>::Description kOccupiedRegionHalfWidthMParamDesc{
    "occupied_region_half_width_m", 0.1f,
    "Half the width of the region which is considered as occupied."};

constexpr Param<float>::Description kMissRayOccupancyProbabilityParamDesc{
    "miss_ray_occupancy_probability", 0.48f,
    "The inverse sensor model occupancy probability applied along a LiDAR "
    "no-return (miss) ray. Only used when lidar_no_return_free_depth_m is "
    "enabled. Must be weaker (closer to 0.5) than "
    "free_region_occupancy_probability: at coarse voxels a thin obstacle "
    "shares a voxel with the sky behind it, so a miss ray carrying the full "
    "free prior erases it."};

struct OccupancyIntegratorParams {
  Param<float> free_region_occupancy_probability{
      kFreeRegionOccupancyProbabilityParamDesc};
  Param<float> occupied_region_occupancy_probability{
      kOccupiedRegionOccupancyProbabilityParamDesc};
  Param<float> unobserved_region_occupancy_probability{
      kUnobservedRegionOccupancyProbabilityParamDesc};
  Param<float> occupied_region_half_width_m{kOccupiedRegionHalfWidthMParamDesc};
  Param<float> miss_ray_occupancy_probability{
      kMissRayOccupancyProbabilityParamDesc};
};
}  // namespace nvblox
