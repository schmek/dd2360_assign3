#include <random>
#include <array>
#include <algorithm>
#include <memory>
#include <fstream>

#include "exercises.cuh"
#include "device_array.cuh"
#include "profiling_timer.cuh"
#include "exercise_2b.cuh"
#include "exercise_2_utility.cuh"

namespace 
{
  template <size_t N_PARTICLES, size_t N_ITER, size_t BLOCK_SZ> 
  class Simulation
  {
  public:

    void run(std::ofstream& gpu_file, std::ofstream& cpu_file, bool run_cpu = true)
    {
      try
      {
        printf("==== Running simulation with %d particles over %d iterations\n\n", N_PARTICLES, N_ITER);
        auto particles = generate_data();
        auto velocities = generate_data();

        const float dt = 1.f;
        
        //GPU
        {
          ProfilingTimer timer("Evolving particles on GPU");

          float3* d_particles;
          float3* h_particles;
          CUDA_CHECK(cudaMallocHost(&h_particles, N_PARTICLES * sizeof(float3)));
          auto M = BLOCK_SZ * ((N_PARTICLES + BLOCK_SZ - 1) / BLOCK_SZ);
          CUDA_CHECK(cudaMalloc(&d_particles, M * sizeof(float3))); //Allocating enough memory to fill out each block. 
          
          DeviceArray<float3, N_PARTICLES, BLOCK_SZ> d_velocities(*velocities);


          for (size_t i = 0; i < N_ITER; ++i)
          {
            CUDA_CHECK(cudaMemcpy(d_particles, h_particles, N_PARTICLES * sizeof(float3), cudaMemcpyHostToDevice));
            CUDA_LAUNCH(evovle_particle_gpu, BLOCK_SZ, N_PARTICLES,
              d_particles, d_velocities.device_get(), dt, 1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_particles, d_particles, N_PARTICLES * sizeof(float3), cudaMemcpyDeviceToHost));
          }
          CUDA_CHECK(cudaFree(d_particles));
          CUDA_CHECK(cudaFreeHost(h_particles));
          
          auto ms = timer.lap_time();
          gpu_file << N_PARTICLES << ", " << N_ITER << ", " << BLOCK_SZ << ", " << ms << '\n';
        }

        // CPU
        if (run_cpu)
        {
          ProfilingTimer timer("Evolving particles on CPU");
          evolve_particle_cpu<N_PARTICLES>(particles->data(), velocities->data(), dt, N_ITER);
          auto ms = timer.lap_time();
          cpu_file << N_PARTICLES << ", " << N_ITER << ", " << ms << '\n';
        }

      }
      catch (const CudaError& ex)
      {
        printf("Cuda error: %s\n source: %s\n", ex.what(), ex.source().c_str());
      }
    }

  private:
    auto generate_data()
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<float> dist(0.f, 1.f);

      auto data = std::make_unique<std::array<float3, N_PARTICLES>>();
      std::generate(data->begin(), data->end(), [&] { return get_random_vector(gen, dist); });
      return data;
    }
  };
}

void exercise_2a()
{
  std::ofstream cpu_file("cpu.csv", std::ios::out);
  std::ofstream gpu_file("gpu.csv", std::ios::out);

  cpu_file << "particles, iterations, time\n";
  gpu_file << "particles, iterations, block_size, time\n";

  Simulation<1'048'576, 8'192, 2 << 8>().run(gpu_file, cpu_file);
  
}

int main(int, char**)
{
  try 
  {
    exercise_2b();
    return EXIT_SUCCESS;
  }
  catch (const CudaError& ex)
  {
    printf("Cuda error: %s\n", ex.what());
  }
  return EXIT_FAILURE;
}

