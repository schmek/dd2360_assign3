#pragma once

void exercise_2b();

template<typename Distribution>
float3 get_random_vector(std::mt19937& gen, Distribution& distribution)
{
  float3 vec;
  vec.x = distribution(gen);
  vec.y = distribution(gen);
  vec.z = distribution(gen);
  return vec;
}

__device__ inline void iterate_particle(float3* particle, float3* velocity, const float dt, const size_t idx)
{
  particle->x += velocity->x * dt;
  particle->y += velocity->y * dt;
  particle->z += velocity->z * dt;
}

__global__ inline void evovle_particle_gpu(float3 particles[], float3 velocities[], const float dt, const size_t n_iter)
{
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  float3* particle = &particles[idx];
  float3* velocity = &velocities[idx];
  for (size_t i = 0; i < n_iter; ++i)
  {
    iterate_particle(particle, velocity, dt, idx);
  }
}

template<size_t ARRAY_SIZE>
__host__ void evolve_particle_cpu(float3 particles[], float3 velocities[], const float dt, const size_t n_iter)
{
  for (size_t j = 0; j < ARRAY_SIZE; ++j)
  {
    for (size_t i = 0; i < n_iter; ++i)
    {
      particles[j].x += velocities[j].x * dt;
      particles[j].y += velocities[j].y * dt;
      particles[j].z += velocities[j].z * dt;
    }
  }
}
