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

#include <limits>

#include "nvblox/integrators/internal/projective_integrator.h"
#include "nvblox/integrators/weighting_function.h"

#include "nvblox/core/log_odds.h"
#include "nvblox/integrators/occupancy_integrator_params.h"

namespace nvblox {

struct UpdateOccupancyVoxelFunctor;

/// A class performing occupancy intregration
///
/// Integrates depth images into occupancy layers. The
/// "projective" describes one type of integration. Namely that voxels in view
/// are projected into the depth image (the alternative being casting rays out
/// from the sensor).
class ProjectiveOccupancyIntegrator
    : public ProjectiveIntegrator<OccupancyVoxel> {
 public:
  ProjectiveOccupancyIntegrator();
  ProjectiveOccupancyIntegrator(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~ProjectiveOccupancyIntegrator();

  /// Integrates a depth image in to the passed occupancy layer.
  /// @param depth_frame A depth image.
  /// @param T_L_C The pose of the sensor. Supplied as a Transform mapping
  /// points in the sensor frame (C) to the layer frame (L).
  /// @param sensor The sensor (intrinsic) model.
  /// @param layer A pointer to the layer into which this observation will
  /// be intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain
  /// the 3D indices of blocks affected by the integration.
  template <typename SensorType>
  void integrateFrame(const MaskedDepthImageConstView& depth_frame,
                      const Transform& T_L_C, const SensorType& sensor,
                      OccupancyLayer* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// A parameter getter
  /// The occupancy probability (inverse sensor model) of the free region
  /// observed on the sensor.
  /// @returns the free region occupancy probability
  float free_region_occupancy_probability() const;

  /// A parameter setter
  /// See free_region_occupancy_probability().
  /// @param value the free region occupancy probability.
  void free_region_occupancy_probability(float value);

  /// A parameter getter
  /// The occupancy probability (inverse sensor model) of the occupied region
  /// observed on the sensor.
  /// @returns the occupied occupancy probability
  float occupied_region_occupancy_probability() const;

  /// A parameter setter
  /// See occupied_region_occupancy_probability().
  /// @param value the occupied occupancy probability
  void occupied_region_occupancy_probability(float value);

  /// A parameter getter
  /// The occupancy probability (inverse sensor model) of the unobserved region
  /// @returns the unobserved occupancy probability
  float unobserved_region_occupancy_probability() const;

  /// A parameter setter
  /// See unobserved_region_occupancy_probability().
  /// @param value the unobserved occupancy probability
  void unobserved_region_occupancy_probability(float value);

  /// A parameter getter
  /// Half the width of the region which is consided as occupied i.e. where
  /// occupied_region_log_odds is applied to all voxels on update. The region is
  /// centered at the measured surface depth on the integrated frame.
  /// @returns the occupied region half width in meters
  float occupied_region_half_width_m() const;

  /// A parameter setter
  /// See occupied_region_half_width_m().
  /// @param occupied_region_half_width_m the occupied region half width in
  /// meters
  void occupied_region_half_width_m(float occupied_region_half_width_m);

  /// A parameter getter
  /// The occupancy probability applied along a LiDAR no-return (miss) ray.
  /// @returns the miss ray occupancy probability
  float miss_ray_occupancy_probability() const;

  /// A parameter setter
  /// See miss_ray_occupancy_probability().
  /// @param value the miss ray occupancy probability
  void miss_ray_occupancy_probability(float value);

  /// A parameter getter
  /// Measured depth at or above which a pixel is treated as a synthetic
  /// no-return sentinel rather than a surface. Infinity disables miss-ray
  /// handling, which is the default.
  /// @returns the miss ray depth threshold in meters
  float miss_ray_min_depth_m() const;

  /// A parameter setter
  /// See miss_ray_min_depth_m(). Pass the sentinel depth written by the
  /// pointcloud-to-depth conversion; a small margin is subtracted internally.
  /// Pass a non-positive value to disable miss-ray handling.
  /// @param no_return_free_depth_m the sentinel depth in meters
  void miss_ray_sentinel_depth_m(float no_return_free_depth_m);

  /// A parameter getter
  /// Range beyond which a no-return (miss) ray stops carving free space. Zero
  /// means carve all the way to the integration limit.
  /// @returns the miss ray carve limit in meters
  float miss_ray_max_carve_distance_m() const;

  /// A parameter setter
  /// See miss_ray_max_carve_distance_m(). Also bounds the raycast used to
  /// select blocks, so a capped miss ray does not allocate empty sky blocks
  /// out to the integration limit.
  /// @param miss_ray_max_carve_distance_m the carve limit in meters
  void miss_ray_max_carve_distance_m(float miss_ray_max_carve_distance_m);

  /// For voxels with a radius, allocate memory and give a small weight and
  /// truncation distance, effectively making these voxels free-space. Does not
  /// affect voxels which are already observed.
  /// @param center The center of the sphere affected.
  /// @param radius The radius of the sphere affected.
  /// @param layer A pointed to the layer which will be affected by the update.
  /// @param updated_blocks_ptr Optional pointer to a list of blocks affected by
  /// the update.
  void markUnobservedFreeInsideRadius(
      const Vector3f& center, float radius, OccupancyLayer* layer,
      std::vector<Index3D>* updated_blocks_ptr = nullptr);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  void setFunctorParameters(const float block_size);
  std::string getIntegratorName() const override;

  // Pushes the miss-ray thresholds into the view calculator so block selection
  // and voxel update agree on how far a miss ray reaches.
  void syncViewCalculatorMissRayLimits();

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
  float miss_ray_max_carve_distance_m_ =
      kMissRayMaxCarveDistanceMParamDesc.default_value;

  // Functor which defines the voxel update operation.
  unified_ptr<UpdateOccupancyVoxelFunctor> update_functor_host_ptr_;

  // Cuda stream
  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox
