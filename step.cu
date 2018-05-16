#include "helper_math.h"
#include "step.h"

//  Arrays
// Grid [MxMxM]
//density
//temperature
//velocity
// Particles [N]
//position
//velocity
//time
//color

//typedef cudaTextureObject_t cudaTextureObject_t;
//typedef surface<void,cudaSurfaceType3D> surface<void,cudaSurfaceType3D>;

__global__ void pstep(float* verts, float* pvels, float* times, float* colors) {
  int I = blockIdx.x*blockDim.x + threadIdx.x;
  int i = I * 3;
  verts[i  ] += pvels[i  ];
  verts[i+1] += pvels[i+1];
  verts[i+2] += pvels[i+2] + 0.003f*(1.0f-times[I]);

  times[I] -= 0.0001f;
  colors[i  ] = sqrt(times[I]);
  colors[i+1] = max(times[I]/1.125f, 0.0f);
  colors[i+2] = pow(times[I],2.0f);
  // move this to the CPU and initialize position randomly within a sphere
  if (times[I] <= 0.0f) {
    times[I]   = 1.0f;
    verts[i  ] = 0.0f;
    verts[i+1] = 0.0f;
    verts[i+2] = 0.0f;
  }
}

__device__ void advect(cudaTextureObject_t vels, cudaTextureObject_t src, surface<void,cudaSurfaceType3D> dest, float dissipation,
                        int I, int J, int K) {
  ////read indices
  //int I = blockIdx.x*blockDim.x + threadIdx.x;
  //int J = blockIdx.y*blockDim.y + threadIdx.y;
  //int K = blockIdx.z*blockDim.z + threadIdx.z;
  ////write index
  //int i = I*blockDim.x*blockDim.y + J*blockDim.y + K;

  float4 v = tex3D((texture<float4, 3, cudaReadModeElementType>)vels, I, J, K);
}

__global__ void gstep(cudaTextureObject_t gvels, cudaTextureObject_t gtemp, cudaTextureObject_t gpres, float* verts, int ping, int pong) {
  //read indices
  int I = blockIdx.x*blockDim.x + threadIdx.x;
  int J = blockIdx.y*blockDim.y + threadIdx.y;
  int K = blockIdx.z*blockDim.z + threadIdx.z;
  //write index
  //int i = I*blockDim.x*blockDim.y + J*blockDim.y + K;

  advect(gvels, gvels, s_gvels[pong], 0.01, I, J, K);

}

void step_gpu(float* verts, float* pvels, float* times, float* colors,
              cudaTextureObject_t gvels, cudaTextureObject_t gtemp, cudaTextureObject_t gpres,
              surface<void,cudaSurfaceType3D> s_gvels, surface<void,cudaSurfaceType3D> s_gtemp, surface<void,cudaSurfaceType3D> s_gpres,
              const int N, const int M, int ping, int pong) {
  //dim3 gblock(M/8,M/8,M/8);
  //dim3 gthread(8,8,8);
  //gstep<<<gblock, gthread>>>(gvels, gtemp, gpres, s_gvels, verts, ping, pong);
  //int pblock = N/32;
  //int pthread = 32;
  //pstep<<<pblock, pthread>>>(verts, pvels, times, colors);
}

void step_cpu(float* verts, float* vels, float* temps, float* colors, int N) {
#pragma omp parallel for
  for (int I=0; I < N; ++I) {
    int i = 3*I;
    verts[i  ] += vels[i  ];
    verts[i+1] += vels[i+1];
    verts[i+2] += vels[i+2] + 0.003*(1.0-temps[I]);

    temps[I] -= 0.0001;
    colors[i  ] = sqrt(temps[I]);
    colors[i+1] = max(temps[I]/1.125, 0.0);
    colors[i+2] = pow(temps[I],2);
    if (temps[I] <= 0.0) {
      temps[I] = 1.0;
      verts[i  ] = 0.0;
      verts[i+1] = 0.0;
      verts[i+2] = 0.0;
    }
  }
}
