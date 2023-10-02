#include "stdGL.h"
#include <vector>
#include <iostream>
#include "objects.h"
#include "shader.h"
#include "helper_math.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
//#include <SDL2/SDL_image.h>

using namespace std;

//GLOBAL VARIABLES//
//running or not
bool quit = false;
int Pause = 0;

//Window Size
int w = 1920;
int h = 1080;

//eye position and orientation
double ex = 0;
double ey = 0;
double ez = 0;
double zoom = 64;
double dzoom = 0;
double th = 0;
double ph = 0;
double dph = 0;
double dth = 0;

//Textures
unsigned int starTexture = 0;

//Shaders
int shader = 0;
int pixlight = 0;
//int textures = 0;
//int test = 0;

//Simulation Timestep
const float dt = 1.0/64.0;
//const float dt = 0.125;
bool changed = false;

//Cuda random state
curandState_t* curandstate;

// Array Sizes
//const int N = pow(2,20);
const int N = 1024*64;
const int M = 64;
int ping = 0;
int pong = 1;

float zeros[M*M*M*4] = {0.0};
float ones[M*M*M]    = {1.0};

//Particle Arrays
float* verts  = NULL;
float* pvels  = NULL;
float* times  = NULL;
float* colors = NULL;

float* dverts = NULL;
float* dpvels = NULL;
float* dtimes = NULL;
float* dcolors= NULL;

//Grid Arrays
float* h_gvels  = NULL;
//float* h_gtemp  = NULL;
//float* h_gdens  = NULL;

float4* d_gvels[2] = {NULL};
//float*  d_gtemp[2] = {NULL};
//float*  d_gdens[2] = {NULL};
//float*  d_gpres[2] = {NULL};
//float*  d_diverge  =  NULL;

//User Interaction
bool rotating = false;
bool flame_moving = false;
bool fieldlines = false;
bool stepmode = false;
bool gpu = true;
float flame_x = M/2.0;
float flame_y = M/2.0;
float tick_period = 4.0;
float decay_rate = 0.008;
//float flame_z = 0.0;

////////////////////
//functions that are called ahead of when they're defined
//because C
void reshape(int width, int height);
void keyboard(const Uint8* state);

///////// CUDA Functions //////////

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

// non-texture-memory texture lookup function
__device__ float4 tex3d(float4* tex, float i, float j, float k, int s_i, int s_j, int s_k) {
  //int r1 = floor(r); r1 = r1%s_r;
  //int r2 = ceil(r);  r2 = r2%s_r;
  //int s1 = floor(s); s1 = s1%s_s;
  //int s2 = ceil(s);  s2 = s2%s_s;
  //int t1 = floor(t); t1 = t1%s_t;
  //int t2 = ceil(t);  t2 = t2%s_t;
  i = clamp(i, 0.0, s_i-1.0);
  j = clamp(j, 0.0, s_j-1.0);
  k = clamp(k, 0.0, s_k-1.0);
  int i1 = floor(i);
  int i2 = ceil (i);
  int j1 = floor(j);
  int j2 = ceil (j);
  int k1 = floor(k);
  int k2 = ceil (k);

  //if (t1 == 0 || t2 == s_t-1)
  //  return 0.0;

  float4 a = tex[i1*s_j*s_k + j1*s_k + k1];
  float4 b = tex[i2*s_j*s_k + j1*s_k + k1];
  float4 c = tex[i1*s_j*s_k + j2*s_k + k1];
  float4 d = tex[i2*s_j*s_k + j2*s_k + k1];
  float4 e = tex[i1*s_j*s_k + j1*s_k + k2];
  float4 f = tex[i2*s_j*s_k + j1*s_k + k2];
  float4 g = tex[i1*s_j*s_k + j2*s_k + k2];
  float4 h = tex[i2*s_j*s_k + j2*s_k + k2];
  return trilerp(a,b,c,d,e,f,g,h, i-i1,j-j1,k-k1);
}
__device__ float tex3d(float* tex, float i, float j, float k, int s_i, int s_j, int s_k) {
  //int r1 = floor(r); r1 = r1%s_r;
  //int r2 = ceil(r);  r2 = r2%s_r;
  //int s1 = floor(s); s1 = s1%s_s;
  //int s2 = ceil(s);  s2 = s2%s_s;
  //int t1 = floor(t); t1 = t1%s_t;
  //int t2 = ceil(t);  t2 = t2%s_t;
  i = clamp(i, 0.0, s_i-1.0);
  j = clamp(j, 0.0, s_j-1.0);
  k = clamp(k, 0.0, s_k-1.0);
  int i1 = floor(i);
  int i2 = ceil (i);
  int j1 = floor(j);
  int j2 = ceil (j);
  int k1 = floor(k);
  int k2 = ceil (k);

  //if (t1 == 0 || t2 == s_t-1)
  //  return 0.0;

  float a = tex[i1*s_j*s_k + j1*s_k + k1];
  float b = tex[i2*s_j*s_k + j1*s_k + k1];
  float c = tex[i1*s_j*s_k + j2*s_k + k1];
  float d = tex[i2*s_j*s_k + j2*s_k + k1];
  float e = tex[i1*s_j*s_k + j1*s_k + k2];
  float f = tex[i2*s_j*s_k + j1*s_k + k2];
  float g = tex[i1*s_j*s_k + j2*s_k + k2];
  float h = tex[i2*s_j*s_k + j2*s_k + k2];
  return trilerp(a,b,c,d,e,f,g,h, i-i1,j-j1,k-k1);
}

__device__ void set_bnd(float4* vels) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;

  // Boundary conditions:
  // X: reflect
  // Y: reflect
  // Z: reflect
  if (i == 0) {
    float4 src = vels[(i+1)*M*M + j*M + k];
    vels[i*M*M + j*M + k] = make_float4(-src.x, src.y, src.z, src.w);
    //vels[i*M*M + j*M + k] = make_float4(0.0, 0.0, 0.0, src.w);
  }
  else if (i == M-1) {
    float4 src = vels[(i-1)*M*M + j*M + k];
    vels[i*M*M + j*M + k] = make_float4(-src.x, src.y, src.z, src.w);
    //vels[i*M*M + j*M + k] = make_float4(0.0, 0.0, 0.0, src.w);
  }
  if (j == 0) {
    float4 src = vels[i*M*M + (j+1)*M + k];
    vels[i*M*M + j*M + k] = make_float4(src.x, -src.y, src.z, src.w);
    //vels[i*M*M + j*M + k] = make_float4(0.0, 0.0, 0.0, src.w);
  }
  else if (j == M-1) {
    float4 src = vels[i*M*M + (j-1)*M + k];
    vels[i*M*M + j*M + k] = make_float4(src.x, -src.y, src.z, src.w);
    //vels[i*M*M + j*M + k] = make_float4(0.0, 0.0, 0.0, src.w);
  }
  if (k == 0) {
    float4 src = vels[i*M*M + j*M + (k+1)];
    vels[i*M*M + j*M + k] = make_float4(src.x, src.y, -src.z, src.w);
    //vels[i*M*M + j*M + k] = make_float4(0.0, 0.0, 0.0, src.w);
  }
  else if (k == M-1) {
    float4 src = vels[i*M*M + j*M + (k-1)];
    vels[i*M*M + j*M + k] = make_float4(src.x, src.y, -src.z, src.w);
    //vels[i*M*M + j*M + k] = make_float4(0.0, 0.0, 0.0, src.w);
  }
}

//__device__ void lin_solv(float4* x, float4* x0, float a, float c) {
//  int i = blockIdx.x*blockDim.x + threadIdx.x;
//  int j = blockIdx.y*blockDim.y + threadIdx.y;
//  int k = blockIdx.z*blockDim.z + threadIdx.z;
//
//// TODO: Enable cooperative syncing iff block-edges become noticable or incompressibility is broken
////  cooperative_groups::grid_group g = cooperative_groups::this_grid();
//
//  float cc = 1.0/c;
//  if (i > 0 && i < M-1 &&
//      j > 0 && j < M-1 &&
//      k > 0 && k < M-1) {
//    for (int iter=0; iter < 16; ++iter) {
//      x[i*M*M + j*M + k] =
//        (x0[i*M*M + j*M + k]
//        + a*(x[(i+1)*M*M + j*M + k] + x[(i-1)*M*M + j*M + k]
//           + x[i*M*M + (j+1)*M + k] + x[i*M*M + (j-1)*M + k]
//           + x[i*M*M + j*M + (k+1)] + x[i*M*M + j*M + (k-1)])
//        )*cc;
//      set_bnd(x);
////      g.sync();
//    }
//  }
//}

__global__ void diffuse(float4* x, float4* x0, float viscosity) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;

  float a = dt * viscosity;// * (M-2)*(M-2)*(M-2);
  float cc = 1.0/(1.0+6.0*a);
  if (i > 0 && i < M-1 &&
      j > 0 && j < M-1 &&
      k > 0 && k < M-1) {
    x[i*M*M + j*M + k] =
      (x0[i*M*M + j*M + k] +
          a*(x0[(i+1)*M*M + j*M + k] + x0[(i-1)*M*M + j*M + k]
           + x0[i*M*M + (j+1)*M + k] + x0[i*M*M + (j-1)*M + k]
           + x0[i*M*M + j*M + (k+1)] + x0[i*M*M + j*M + (k-1)])
      )*cc;
  }
  set_bnd(x);
}

__global__ void pressure(float4* vels, float4* vels0) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;

  if (i > 0 && i < M-1 &&
      j > 0 && j < M-1 &&
      k > 0 && k < M-1) {
    // collect neighboring densities
    float p_x0 = vels0[(i-1)*M*M + j*M + k].w;
    float p_x1 = vels0[(i+1)*M*M + j*M + k].w;
    float p_y0 = vels0[i*M*M + (j-1)*M + k].w;
    float p_y1 = vels0[i*M*M + (j+1)*M + k].w;
    float p_z0 = vels0[i*M*M + j*M + (k-1)].w;
    float p_z1 = vels0[i*M*M + j*M + (k+1)].w;
    // collect neighboring velocities
    float v_x0 = vels0[(i-1)*M*M + j*M + k].x;
    float v_x1 = vels0[(i+1)*M*M + j*M + k].x;
    float v_y0 = vels0[i*M*M + (j-1)*M + k].y;
    float v_y1 = vels0[i*M*M + (j+1)*M + k].y;
    float v_z0 = vels0[i*M*M + j*M + (k-1)].z;
    float v_z1 = vels0[i*M*M + j*M + (k+1)].z;
    // apply net pressure force
    float d_x = 5.0*(p_x0 - p_x1);
    float d_y = 5.0*(p_y0 - p_y1);
    float d_z = 5.0*(p_z0 - p_z1);
    // and add vertical buoyancy force
    //float p_b = vels0[i*M*M + j*M + k].w - 0.16666*(p_x0 + p_x1 + p_y0 + p_y1 + p_z0 + p_z1);
    //float p_b = 0.0;

    //float buoy = 1.0;
    //float a = dt;
    //float a = 5.0;

    // modify pressure based on net velocity
    float d_p = 1.0 * (v_x0 - v_x1
                     + v_y0 - v_y1
                     + v_z0 - v_z1);

    vels[i*M*M + j*M + k].x = vels0[i*M*M + j*M + k].x + dt*d_x;
    vels[i*M*M + j*M + k].y = vels0[i*M*M + j*M + k].y + dt*d_y;
    vels[i*M*M + j*M + k].z = vels0[i*M*M + j*M + k].z + dt*d_z;
    // small velocity decay to naturally trend back toward zero instead of infinity over time
    vels[i*M*M + j*M + k] *= 0.999;
    vels[i*M*M + j*M + k].w = d_p;
  }

  set_bnd(vels);
}

__global__ void project(float4* vels) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;

  // find divergences of adjacent cells
  float dx0 = 0.0;
  if (i > 1 && j > 0 && j < M-1 && k > 0 && k < M-1) {
    dx0 = 0.16666*( - vels[(i-2)*M*M + j*M + k].x
                    + vels[i*M*M + j*M + k].x
                    - vels[(i-1)*M*M + (j-1)*M + k].y
                    + vels[(i-1)*M*M + (j+1)*M + k].y
                    - vels[(i-1)*M*M + j*M + (k-1)].z
                    + vels[(i-1)*M*M + j*M + (k+1)].z);
  }
  float dx1 = 0.0;
  if (i < M-2 && j > 0 && j < M-1 && k > 0 && k < M-1) {
    dx1 = 0.16666*( - vels[i*M*M + j*M + k].x
                    + vels[(i+2)*M*M + j*M + k].x
                    - vels[(i+1)*M*M + (j-1)*M + k].y
                    + vels[(i+1)*M*M + (j+1)*M + k].y
                    - vels[(i+1)*M*M + j*M + (k-1)].z
                    + vels[(i+1)*M*M + j*M + (k+1)].z);
  }
  float dy0 = 0.0;
  if (i > 0 && i < M-1 && j > 1 && k > 0 && k < M-1) {
    dy0 = 0.16666*( - vels[(i-1)*M*M + (j-1)*M + k].x
                    + vels[(i+1)*M*M + (j-1)*M + k].x
                    - vels[i*M*M + (j-2)*M + k].y
                    + vels[i*M*M + j*M + k].y
                    - vels[i*M*M + (j-1)*M + (k-1)].z
                    + vels[i*M*M + (j-1)*M + (k+1)].z);
  }
  float dy1 = 0.0;
  if (i > 0 && i < M-1 && j < M-2 && k > 0 && k < M-1) {
    dy1 = 0.16666*( - vels[(i-1)*M*M + (j+1)*M + k].x
                    + vels[(i+1)*M*M + (j+1)*M + k].x
                    - vels[i*M*M + j*M + k].y
                    + vels[i*M*M + (j+2)*M + k].y
                    - vels[i*M*M + (j+1)*M + (k-1)].z
                    + vels[i*M*M + (j+1)*M + (k+1)].z);
  }
  float dz0 = 0.0;
  if (i > 0 && i < M-1 && j > 0 && j < M-1 && k > 1) {
    dz0 = 0.16666*( - vels[(i-1)*M*M + j*M + (k-1)].x
                    + vels[(i+1)*M*M + j*M + (k-1)].x - vels[i*M*M + (j-1)*M + (k-1)].y
                    + vels[i*M*M + (j+1)*M + (k-1)].y
                    - vels[i*M*M + j*M + (k-2)].z
                    + vels[i*M*M + j*M + k].z);
  }
  float dz1 = 0.0;
  if (i > 0 && i < M-1 && j > 0 && j < M-1 && k < M-2) {
    dz1 = 0.16666*( - vels[(i-1)*M*M + j*M + (k+1)].x
                    + vels[(i+1)*M*M + j*M + (k+1)].x
                    - vels[i*M*M + (j-1)*M + (k+1)].y
                    + vels[i*M*M + (j+1)*M + (k+1)].y
                    - vels[i*M*M + j*M + k].z
                    + vels[i*M*M + j*M + (k+2)].z);
  }

  // subtract pressure vectors from velocities
  vels[i*M*M + j*M + k].x -= 0.5*(dx1 - dx0);
  vels[i*M*M + j*M + k].y -= 0.5*(dy1 - dy0);
  vels[i*M*M + j*M + k].z -= 0.5*(dz1 - dz0);

  set_bnd(vels);
}

__global__ void balance(float4* vels) {//, float4* vels0) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;

  //float sum_pres = 100.0;
  float sum_pres = 1.0;

  //for (int I=1; I < M*M*M; ++I) {
  //  sum_pres += vels0[I].w;
  //}
  sum_pres = sum_pres / (M*M*M);
  vels[i*M*M + j*M + k].w -= sum_pres;

  set_bnd(vels);
}

__global__ void advect(float4* vels_out, float4* vels_in, float* verts, float* times) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  int k = blockIdx.z*blockDim.z + threadIdx.z;

  float fi = i - dt*vels_in[i*M*M + j*M + k].x;
  float fj = j - dt*vels_in[i*M*M + j*M + k].y;
  float fk = k - dt*vels_in[i*M*M + j*M + k].z;

  vels_out[i*M*M + j*M + k] = tex3d(vels_in, fi, fj, fk, M,M,M);

  //// Add impulse based on heat buoyancy from the particles
  //float impulse = 0.0;
  //for (int n=0; n<N; ++n) {
  //  float3 vert = make_float3(verts[n*3], verts[n*3+1], verts[n*3+2]);
  //  float dx = vert.x-i;
  //  float dy = vert.y-j;
  //  float dz = vert.z-k;
  //  if (abs(dx) < 1.0 && abs(dy) < 1.0 && abs(dz) < 1.0) {
  //    float dist = sqrt(dx*dx + dy*dy + dz*dz);
  //    impulse += (1.0-dist) * times[n];
  //  }
  //}
  //vels_out[i*M*M + j*M + k].z += 4.0*impulse/N;

  //if (abs(i - flame_x) <= 1.0 && abs(j - flame_y) <= 1.0 && k/2 == M/8) {
  //  vels_out[i*M*M + j*M + k].x = 0.0;
  //  vels_out[i*M*M + j*M + k].y = 0.0;
  //  vels_out[i*M*M + j*M + k].z = 0.5;
  //}

  set_bnd(vels_out);
}

__global__ void pingpong(float4* x, float4* x0) {
  // this has better performance than memcpy
  int I = blockIdx.x*blockDim.x + threadIdx.x;
  if (I < M*M*M) x[I] = x0[I];
}

__global__ void init_rand_state(curandState_t* curandstate) {
  int I = blockIdx.x*blockDim.x + threadIdx.x;
  curand_init(1729, I, 0, &curandstate[I]);
}

__global__ void pstep(float4* gvels, float* verts, float* times, float* colors, curandState_t* curandstate, float flame_x, float flame_y, float decay_rate) {
  // times index
  int I = blockIdx.x*blockDim.x + threadIdx.x;
  // verts & colors index
  int i = I * 3;
  // texture lookup of velocity at the particle's location
  float4 V = tex3d(gvels, verts[i], verts[i+1], verts[i+2], M,M,M);
  verts[i  ] += V.x;
  verts[i+1] += V.y;
  verts[i+2] += V.z;

  colors[i  ] = sqrt(times[I]);
  colors[i+1] = max(times[I]/1.125f, 0.0f);
  colors[i+2] = pow(times[I],2.0f)/2;
  //colors[i  ] = max(0.2, abs(V.x));
  //colors[i+1] = max(0.2, abs(V.y));
  //colors[i+2] = max(0.2, abs(V.z));
  times[I] -= decay_rate;
  if (times[I] < 0.0f) {
    times[I]  += 1.0f;
    curandState_t localstate0 = curandstate[I+0];
    curandState_t localstate1 = curandstate[I+1];
    curandState_t localstate2 = curandstate[I+2];
    verts[i  ] = (curand_normal(&localstate0)) + flame_x;
    verts[i+1] = (curand_normal(&localstate1)) + flame_y;
    verts[i+2] = (curand_normal(&localstate2)) + M/4;
    curandstate[I+0] = localstate0;
    curandstate[I+1] = localstate1;
    curandstate[I+2] = localstate2;
  }

  // Add impulse to gvels based on particle temperature
  //    float dist = sqrt(dx*dx + dy*dy + dz*dz);
  //    impulse += (1.0-dist) * times[n];
  //  }
  //}
  //vels_out[i*M*M + j*M + k].z += 4.0*impulse/N;
  //vels_out[i*M*M + j*M + k].w += 1.0*impulse/N;
  float3 pos = make_float3(verts[i], verts[i+1], verts[i+2]);
  if (pos.x > 0 && pos.x < M-1 &&
      pos.y > 0 && pos.y < M-1 &&
      pos.z > 0 && pos.z < M-1) {
    // for each of 8 nearest grid cells
    int i0,j0,k0,i1,j1,k1;
    float share;
    float d_buoy = 1024.0;
    float d_pres = 8192.0;
    // 0,0,0
    i0 = floor(pos.x); i1 = ceil(pos.x);
    j0 = floor(pos.y); j1 = ceil(pos.y);
    k0 = floor(pos.z); k1 = ceil(pos.z);
    // proportion given to each cell is equal to volume of the opposite quadrant
    share = abs((i1-pos.x) * (j1-pos.y) * (k1-pos.z));
    gvels[i0*M*M + j0*M + k0].z += d_buoy*dt*(times[i]-0.0)*share/N;
    gvels[i0*M*M + j0*M + k0].w += d_pres*dt*(times[i]-0.2)*share/N;
    // 0,0,1
    i0 = floor(pos.x); i1 = ceil(pos.x);
    j0 = floor(pos.y); j1 = ceil(pos.y);
    k1 = floor(pos.z); k0 = ceil(pos.z);
    // proportion given to each cell is equal to volume of the opposite quadrant
    share = abs((i1-pos.x) * (j1-pos.y) * (k1-pos.z));
    gvels[i0*M*M + j0*M + k0].z += d_buoy*dt*(times[i]-0.0)*share/N;
    gvels[i0*M*M + j0*M + k0].w += d_pres*dt*(times[i]-0.2)*share/N;
    // 0,1,0
    i0 = floor(pos.x); i1 = ceil(pos.x);
    j1 = floor(pos.y); j0 = ceil(pos.y);
    k0 = floor(pos.z); k1 = ceil(pos.z);
    // proportion given to each cell is equal to volume of the opposite quadrant
    share = abs((i1-pos.x) * (j1-pos.y) * (k1-pos.z));
    gvels[i0*M*M + j0*M + k0].z += d_buoy*dt*(times[i]-0.0)*share/N;
    gvels[i0*M*M + j0*M + k0].w += d_pres*dt*(times[i]-0.2)*share/N;
    // 0,1,1
    i0 = floor(pos.x); i1 = ceil(pos.x);
    j1 = floor(pos.y); j0 = ceil(pos.y);
    k1 = floor(pos.z); k0 = ceil(pos.z);
    // proportion given to each cell is equal to volume of the opposite quadrant
    share = abs((i1-pos.x) * (j1-pos.y) * (k1-pos.z));
    gvels[i0*M*M + j0*M + k0].z += d_buoy*dt*(times[i]-0.0)*share/N;
    gvels[i0*M*M + j0*M + k0].w += d_pres*dt*(times[i]-0.2)*share/N;
    // 1,0,0
    i1 = floor(pos.x); i0 = ceil(pos.x);
    j0 = floor(pos.y); j1 = ceil(pos.y);
    k0 = floor(pos.z); k1 = ceil(pos.z);
    // proportion given to each cell is equal to volume of the opposite quadrant
    share = abs((i1-pos.x) * (j1-pos.y) * (k1-pos.z));
    gvels[i0*M*M + j0*M + k0].z += d_buoy*dt*(times[i]-0.0)*share/N;
    gvels[i0*M*M + j0*M + k0].w += d_pres*dt*(times[i]-0.2)*share/N;
    // 1,0,1
    i1 = floor(pos.x); i0 = ceil(pos.x);
    j0 = floor(pos.y); j1 = ceil(pos.y);
    k1 = floor(pos.z); k0 = ceil(pos.z);
    // proportion given to each cell is equal to volume of the opposite quadrant
    share = abs((i1-pos.x) * (j1-pos.y) * (k1-pos.z));
    gvels[i0*M*M + j0*M + k0].z += d_buoy*dt*(times[i]-0.0)*share/N;
    gvels[i0*M*M + j0*M + k0].w += d_pres*dt*(times[i]-0.2)*share/N;
    // 1,1,0
    i1 = floor(pos.x); i0 = ceil(pos.x);
    j1 = floor(pos.y); j0 = ceil(pos.y);
    k0 = floor(pos.z); k1 = ceil(pos.z);
    // proportion given to each cell is equal to volume of the opposite quadrant
    share = abs((i1-pos.x) * (j1-pos.y) * (k1-pos.z));
    gvels[i0*M*M + j0*M + k0].z += d_buoy*dt*(times[i]-0.0)*share/N;
    gvels[i0*M*M + j0*M + k0].w += d_pres*dt*(times[i]-0.2)*share/N;
    // 1,1,1
    i1 = floor(pos.x); i0 = ceil(pos.x);
    j1 = floor(pos.y); j0 = ceil(pos.y);
    k1 = floor(pos.z); k0 = ceil(pos.z);
    // proportion given to each cell is equal to volume of the opposite quadrant
    share = abs((i1-pos.x) * (j1-pos.y) * (k1-pos.z));
    gvels[i0*M*M + j0*M + k0].z += d_buoy*dt*(times[i]-0.0)*share/N;
    gvels[i0*M*M + j0*M + k0].w += d_pres*dt*(times[i]-0.2)*share/N;
  }
}

void step_gpu(float* verts, float* times, float* colors,
              float4* gvel0, float4* gvel1, //float* gpres0, float* gpres1,
              const int N, const int M, int t,
              float flame_x, float flame_y) {
  int b = 8;
  dim3 gBlock(M/b,M/b,M/b);
  dim3 gThread(b,b,b);

  float visc = 0.10;
  // diffuse velocities
  diffuse<<<gBlock,gThread>>>(gvel1, gvel0, visc);
  //void** diffuse_args[3];
  //diffuse_args[0] = (void**)&gvel1; diffuse_args[1] = (void**)&gvel0; diffuse_args[2] = (void**)&visc;
  //cudaLaunchCooperativeKernel((void*)diffuse, gBlock, gThread, (void**)diffuse_args);
  // Project
  //project<<<gBlock,gThread>>>(gvel1);
  // Pressure
  pressure<<<gBlock,gThread>>>(gvel0, gvel1);
  // Balance pressure
  //balance<<<gBlock,gThread>>>(gvel1);//, gvel1);
  // Advect Velocities
  advect<<<gBlock,gThread>>>(gvel1, gvel0, verts, times);
  // Ping the Pong
  //pingpong<<<Mblocks,512>>>(gvel1, gvel0);
  // Move Particles
  pstep<<<N/512,512>>>(gvel1, verts, times, colors, curandstate, flame_x, flame_y, decay_rate);
}

void step_cpu(float* verts, float* vels, float* times, float* colors, int N) {
#pragma omp parallel for
  for (int I=0; I < N; ++I) {
    int i = 3*I;
    verts[i  ] += vels[i  ];
    verts[i+1] += vels[i+1];
    verts[i+2] += vels[i+2] + 0.003*(1.0-times[I]);

    times[I] -= 0.0001;
    colors[i  ] = sqrt(times[I]);
    colors[i+1] = max(times[I]/1.125, 0.0);
    colors[i+2] = pow(times[I],2);
    if (times[I] <= 0.0) {
      times[I] = 1.0;
      verts[i  ] = M/2;
      verts[i+1] = M/2;
      verts[i+2] = M/2;
    }
  }
}
//////// SDL Init Function ////////

bool init(SDL_Window** window, SDL_GLContext* context)
{
  bool success = true;

  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
  {
    cerr << "SDL failed to initialize: " << SDL_GetError() << endl;
    success = false;
  }

  *window = SDL_CreateWindow("Flame", 0,0, w,h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
  if (*window == NULL)
  {
    cerr << "SDL failed to create a window: " << SDL_GetError() << endl;
    success = false;
  }

  *context = SDL_GL_CreateContext(*window);
  if (*context == NULL)
  {
    cerr << "SDL failed to create OpenGL context: " << SDL_GetError() << endl;
    success = false;
  }

  //Vsync
  if (SDL_GL_SetSwapInterval(1) < 0)
  {
    cerr << "SDL could not set Vsync: " << SDL_GetError() << endl;
//    success = false;
  }

  cout << SDL_GetError() << endl;
  return success;
}

///////////////////////////////////

void display(SDL_Window* window, int r)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  //glEnable(GL_CULL_FACE);

  //reshape(w,h);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  //view angle
  ex = Sin(-th)*Cos(ph)*zoom;
  ey = Cos(-th)*Cos(ph)*zoom;
  ez = Sin(ph)*zoom;

  gluLookAt(ex+M/2,ey+M/2,ez+M/2, M/2,M/2,M/2, 0,0,Cos(ph));

  // lighting
  glEnable(GL_LIGHTING);
  float white[4]   = {1.0,1.0,1.0,1.0};
  float pos[4]     = {M/2+2.0, M/2-2.0, M/2+4.0, 1.0};
  float ambient[4] = {0.12, 0.15, 0.16, 1.0};
  float diffuse[4] = {0.65, 0.65, 0.60, 1.0};
  float specular[4]= {0.7, 0.7, 0.9, 1.0};
  float shininess  = 64;

  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glEnable(GL_COLOR_MATERIAL);

  glEnable(GL_LIGHT0);
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
  glLightfv(GL_LIGHT0, GL_POSITION, pos);

  glMaterialfv(GL_FRONT, GL_SHININESS, &shininess);
  glMaterialfv(GL_FRONT, GL_SPECULAR, white);

  // Object Rendering

  //glUseProgram(pixlight);
  //glColor3f(1.0,1.0,1.0);
  //ball(M/2,M/2,M/2, 0.25);

  if (changed) {
    if(cudaSuccess != cudaMemcpyAsync(verts, dverts, 3*N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail from " << dverts << " to " << verts << "\n";
    if(cudaSuccess != cudaMemcpyAsync(times, dtimes,   N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail from " << dtimes << " to " << times << "\n";
    if(cudaSuccess != cudaMemcpyAsync(colors,dcolors,3*N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail from " << dcolors << " to " << colors << "\n";
  }

  glUseProgram(shader);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glBindTexture(GL_TEXTURE_2D, starTexture);
  int id = glGetUniformLocation(shader, "star");
  if (id>=0) glUniform1i(id,0);
  // ^ current bound texture, star.bmp
  id = glGetUniformLocation(shader, "size");
  if (id>=0) glUniform1f(id,1.0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE,GL_ONE);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glVertexPointer(3,GL_FLOAT,0,verts);
  glColorPointer(3,GL_FLOAT,0,colors);

  //cout << "verts: " << verts[0] << "   \t" << verts[1] << "   \t" << verts[2] << endl;
  //cout << "color: " << colors[0]<< "   \t" << colors[1]<< "   \t" << colors[2] << endl;

  glDrawArrays(GL_POINTS,0,N);

  glDisable(GL_BLEND);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  // Bounding Box
  glUseProgram(0);
  glEnable(GL_DEPTH_TEST);
  glBegin(GL_LINES);
  glColor3f(1.0,1.0,1.0);
  glVertex3f(0,0,0); glVertex3f(0,0,M);
  glVertex3f(0,0,M); glVertex3f(0,M,M);
  glVertex3f(0,M,M); glVertex3f(0,M,0);
  glVertex3f(0,M,0); glVertex3f(0,0,0);
  glVertex3f(0,0,0); glVertex3f(M,0,0);
  glVertex3f(0,0,M); glVertex3f(M,0,M);
  glVertex3f(0,M,M); glVertex3f(M,M,M);
  glVertex3f(0,M,0); glVertex3f(M,M,0);
  glVertex3f(M,0,0); glVertex3f(M,0,M);
  glVertex3f(M,0,M); glVertex3f(M,M,M);
  glVertex3f(M,M,M); glVertex3f(M,M,0);
  glVertex3f(M,M,0); glVertex3f(M,0,0);
  glEnd();
  glDisable(GL_DEPTH_TEST);


  //// DEBUG ////

  if (fieldlines) {
    // show velocities for debug purposes
    if (changed) {
      cudaError_t err = cudaMemcpyAsync(h_gvels, d_gvels[ping], 4*M*M*M*sizeof(float), cudaMemcpyDeviceToHost); if (err != cudaSuccess) {cout << "cudaMemcpy failed: " << cudaGetErrorString(err) << endl; quit = true;}
    }
    //cout << "Successfully copied Velocities from Device to Host\n";
    glUseProgram(0);
    glEnable(GL_DEPTH_TEST);
    glBegin(GL_LINES);
    for (int i=0; i < M; ++i) {
      for (int j=0; j < M; ++j) {
        for (int k=0; k < M; ++k) {
          glColor3f(1.0,0.5,0.0);
          glVertex3f(i, j, k);
          float x = h_gvels[4*(i*M*M + j*M + k)  ]*10.0;
          float y = h_gvels[4*(i*M*M + j*M + k)+1]*10.0;
          float z = h_gvels[4*(i*M*M + j*M + k)+2]*10.0;
          //float x = 0.0;
          //float y = 0.0;
          //float z = h_gvels[4*(i*M*M + j*M + k)+3]*10.0;
          glColor3f(0.5,0.0,0.0);
          glVertex3f(i+x, j+y, k+z);
        }
      }
    }
    glEnd();
    glDisable(GL_DEPTH_TEST);
  }

  //// show other values for debug purposes
  //cudaError_t err = cudaMemcpy(h_gtemp, d_gtemp[ping], M*M*M*sizeof(float), cudaMemcpyDeviceToHost); if (err != cudaSuccess) {cout << "cudaMemcpy failed: " << cudaGetErrorString(err) << endl; quit = true;}
  //glUseProgram(0);
  //glBegin(GL_LINES);
  //for (int i=0; i < M; ++i) {
  //  for (int j=0; j < M; ++j) {
  //    for (int k=0; k < M; ++k) {
  //      glColor3f(1.0,1.0,1.0);
  //      glVertex3f(i, j, k);
  //      float z = h_gtemp[i*M*M + j*M + k]*10.0;
  //      glColor3f(0.1,0.1,0.1);
  //      glVertex3f(i, j, k+z);
  //    }
  //  }
  //}
  //glEnd();

  changed = false;

  //swap the buffers
  glFlush();
  SDL_GL_SwapWindow(window);
}

void physics(int r)
{
  const Uint8* state = SDL_GetKeyboardState(NULL);
  keyboard(state);

  //adjust the eye position
  th += dth;
  ph += dph;
  zoom = zoom<2.0?2.0:zoom+dzoom;

  // Step Flame Animation ////
  if (!stepmode && !Pause) {
    if (gpu) {
      if(cudaSuccess != cudaMemcpyAsync(verts, dverts, 3*N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail from " << dverts << " to " << verts << "\n";
      if(cudaSuccess != cudaMemcpyAsync(times, dtimes,   N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail from " << dtimes << " to " << times << "\n";
      if(cudaSuccess != cudaMemcpyAsync(colors,dcolors,3*N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail from " << dcolors << " to " << colors << "\n";
      ////cout << "successfully copied Particles from Device to Host" << endl;
      //for (int I=0; I < N; ++I) {
      //  int i = I*3;
      //  if (times[I]   < 0.0 ||
      //  verts[i]   < 0.0 ||
      //  verts[i]   > M   ||
      //  verts[i+1] < 0.0 ||
      //  verts[i+1] > M   ||
      //  verts[i+2] < 0.0 ||
      //  verts[i+2] > M  ) {
      //    times[I]   = 1.0f;
      //    verts[i  ] = 8*((float)rand()/(float)RAND_MAX-0.5) + M/2;
      //    verts[i+1] = 8*((float)rand()/(float)RAND_MAX-0.5) + M/2;
      //    verts[i+2] = 8*((float)rand()/(float)RAND_MAX-0.5) + M/2;
      //  }
      //}
      //if (true) { //{*/r < 10000) {
      //  if(cudaSuccess != cudaMemcpy(h_gvels, d_gvels[ping], 4*M*M*M*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail from " << dcolors << " to " << colors << "\n";
      //  h_gvels[4*((M/2)*M*M + (M/2)*M + (M/2))+2] = 1.0;
      //  //h_gvels[4*((M/2)*M*M + (M/2)*M + (M/2))+2] += 1.0;
      //  //h_gvels[4*((M/2-1)*M*M + (M/2)*M + (M/2))+0] = -1.0;
      //  //h_gvels[4*((M/2+1)*M*M + (M/2)*M + (M/2))+0] = 1.0;
      //  //h_gvels[4*((M/2)*M*M + (M/2-1)*M + (M/2))+1] = -1.0;
      //  //h_gvels[4*((M/2)*M*M + (M/2+1)*M + (M/2))+1] = 1.0;
      //  //h_gvels[4*((M/2)*M*M + (M/2)*M + (M/2-1))+2] = -1.0;
      //  //h_gvels[4*((M/2)*M*M + (M/2)*M + (M/2+1))+2] = 1.0;
      //  if(cudaSuccess != cudaMemcpy(d_gvels[ping], h_gvels, 4*M*M*M*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail from " << verts << " to " << dverts << "\n";
      //}
      ////if(cudaSuccess != cudaMemcpy(d_gpres[0], zeros,   M*M*M*sizeof(float), cudaMemcpyHostToDevice)) cout << "failure to memcpy: " << endl;
      ////if(cudaSuccess != cudaMemcpy(d_gpres[1], zeros,   M*M*M*sizeof(float), cudaMemcpyHostToDevice)) cout << "failure to memcpy: " << endl;
      //if(cudaSuccess != cudaMemcpy(dverts, verts, 3*N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail from " << verts << " to " << dverts << "\n";
      //if(cudaSuccess != cudaMemcpy(dtimes, times,   N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail from " << times << " to " << dtimes << "\n";
      //if(cudaSuccess != cudaMemcpy(dcolors,colors,3*N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail from " << colors << " to " << dcolors << "\n";
      ////cout << "successfully copied Particles from Host to Device" << endl;
      step_gpu(dverts, dtimes, dcolors,
               d_gvels[ping], d_gvels[pong],// d_gtemp[ping], d_gtemp[pong], d_gdens[ping], d_gdens[pong], d_gpres[0], d_gpres[1], d_diverge,
               //s_gvels[pong], s_gtemp[pong], s_gdens[pong],
               N, M, r, flame_x, flame_y);
      ping = pong;
      pong = 1-pong;
    }
    else {
      step_cpu(verts, pvels, times, colors, N);
    }
    changed = true;
  }
  ////////////////////////////
}


void reshape(int width, int height)
{
  w = width;
  h = height;
  //new aspect ratio
  double w2h = (height > 0) ? (double)width/height : 1;
  //set viewport to the new window
  glViewport(0,0 , width,height);

  //switch to projection matrix
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  //adjust projection
  //glOrtho(-w2h, w2h, -1, 1, -1, 1);
  gluPerspective(60, w2h, 1.0, 4*M);

  //switch back to model matrix
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

static void Reverse(void* x,const int n)
{
   int k;
   char* ch = (char*)x;
   for (k=0;k<n/2;k++)
   {
      char tmp = ch[k];
      ch[k] = ch[n-1-k];
      ch[n-1-k] = tmp;
   }
}
int LoadTexture(const char* file) {
  unsigned int   texture;    // Texture name
  FILE*          f;          // File pointer
  unsigned short magic;      // Image magic
  int            dx,dy;
  unsigned int   size;       // Image dimensions
  unsigned short nbp,bpp;    // Planes and bits per pixel
  unsigned char* image;      // Image data
  unsigned int   k;          // Counter
  int            max;        // Maximum texture dimensions

  //  Open file
  f = fopen(file,"rb");
  if (!f) fprintf(stderr,"Cannot open file %s\n",file);
  //  Check image magic
  if (fread(&magic,2,1,f)!=1) fprintf(stderr,"Cannot read magic from %s\n",file);
  if (magic!=0x4D42 && magic!=0x424D) fprintf(stderr,"Image magic not BMP in %s\n",file);
  //  Seek to and read header
  if (fseek(f,16,SEEK_CUR) || fread(&dx ,4,1,f)!=1 || fread(&dy ,4,1,f)!=1 ||
      fread(&nbp,2,1,f)!=1 || fread(&bpp,2,1,f)!=1 || fread(&k,4,1,f)!=1)
    fprintf(stderr,"Cannot read header from %s\n",file);
  //  Reverse bytes on big endian hardware (detected by backwards magic)
  if (magic==0x424D)
  {
     Reverse(&dx,4);
     Reverse(&dy,4);
     Reverse(&nbp,2);
     Reverse(&bpp,2);
     Reverse(&k,4);
  }

  dx = abs(dx);
  dy = abs(dy);

  //  Check image parameters
  glGetIntegerv(GL_MAX_TEXTURE_SIZE,&max);
  if (dx<1 || dx>max) fprintf(stderr,"%s image width %d out of range 1-%d\n",file,dx,max);
  if (dy<1 || dy>max) fprintf(stderr,"%s image height %d out of range 1-%d\n",file,dy,max);
  if (nbp!=1)  fprintf(stderr,"%s bit planes is not 1: %d\n",file,nbp);
  if (bpp!=24) fprintf(stderr,"%s bits per pixel is not 24: %d\n",file,bpp);
  if (k!=0)    fprintf(stderr,"%s comdenssed files not supported\n",file);
#ifndef GL_VERSION_2_0
  //  OpenGL 2.0 lifts the restriction that texture size must be a power of two
  for (k=1;k<dx;k*=2);
  if (k!=dx) fprintf(stderr,"%s image width not a power of two: %d\n",file,dx);
  for (k=1;k<dy;k*=2);
  if (k!=dy) fprintf(stderr,"%s image height not a power of two: %d\n",file,dy);
#endif

  //  Allocate image memory
  size = 3*dx*dy;
  image = (unsigned char*) malloc(size);
  if (!image) fprintf(stderr,"Cannot allocate %d bytes of memory for image %s\n",size,file);
  //  Seek to and read image
  if (fseek(f,20,SEEK_CUR) || fread(image,size,1,f)!=1) fprintf(stderr,"Error reading data from image %s\n",file);
  fclose(f);
  //  Reverse pvels   (BGR -> RGB)
  for (k=0;k<size;k+=3)
  {
     unsigned char temp = image[k];
     image[k]   = image[k+2];
     image[k+2] = temp;
  }

  //  Sanity check
  //ErrCheck("LoadTexBMP");
  //  Generate 2D texture
  glGenTextures(1,&texture);
  glBindTexture(GL_TEXTURE_2D,texture);
  //  Copy image
  glTexImage2D(GL_TEXTURE_2D,0,3,dx,dy,0,GL_RGB,GL_UNSIGNED_BYTE,image);
  if (glGetError()) fprintf(stderr,"Error in glTexImage2D %s %dx%d\n",file,dx,dy);
  //  Scale linearly when image size doesn't match
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);

  //  Free image memory
  free(image);
  //  Return texture name
  return texture;

}

// Per frame keyboard input here, per keydenss input in main()
void keyboard(const Uint8* state)
{
  //if (state[SDL_SCANCODE_ESCAPE])
  //  quit = true;

  if (state[SDL_SCANCODE_LEFT])
    dth = -0.75;
  else if (state[SDL_SCANCODE_RIGHT])
    dth = 0.75;
  else
    dth = 0;

  if (state[SDL_SCANCODE_DOWN])
    dph = -0.75;
  else if (state[SDL_SCANCODE_UP])
    dph = 0.75;
  else
    dph = 0;

  if (state[SDL_SCANCODE_Z])
    dzoom = -0.10;
  else if (state[SDL_SCANCODE_X])
    dzoom = 0.10;
  else
    dzoom = 0;
}

// all user interaction goes here
bool handleEvents()
{
  SDL_Event event;

  while (SDL_PollEvent(&event))
  {
    switch(event.type)
    {
      case SDL_QUIT:
        return true;

      case SDL_MOUSEMOTION:
        if (flame_moving)
        {
          float dx_f = ( Cos(th)*event.motion.xrel + Sin(th)*event.motion.yrel) / M * 2.0;
          float dy_f = (-Sin(th)*event.motion.xrel + Cos(th)*event.motion.yrel) / M * 2.0;
          flame_x = min(max(flame_x - dx_f, 1.0), M-2.0);
          flame_y = min(max(flame_y + dy_f, 1.0), M-2.0);
          //cout << flame_x << "\t" << flame_y << endl;
        }
        if (rotating)
        {
          th -= event.motion.xrel/(2.0*M_PI);
          ph += event.motion.yrel/(2.0*M_PI);
        }
        break;

      case SDL_MOUSEWHEEL:
        if (event.wheel.y > 0)
          zoom /= pow(2.0, 1.0/8.0);
        else if (event.wheel.y < 0)
          zoom *= pow(2.0, 1.0/8.0);
        break;

      case SDL_MOUSEBUTTONDOWN:
        switch (event.button.button)
        {
          case SDL_BUTTON_LEFT:
            flame_moving = true;
            break;

          case SDL_BUTTON_RIGHT:
            rotating = true;
            break;
        }
        break;

      case SDL_MOUSEBUTTONUP:
        switch (event.button.button)
        {
          case SDL_BUTTON_LEFT:
            flame_moving = false;
            break;

          case SDL_BUTTON_RIGHT:
            rotating = false;
            break;
        }
        break;

      case SDL_KEYDOWN:
        switch (event.key.keysym.scancode)
        {
          case SDL_SCANCODE_Q:
            return true;

          case SDL_SCANCODE_SPACE:
            Pause = 1 - Pause;
            break;

          case SDL_SCANCODE_M:
            stepmode = !stepmode;
            break;

          case SDL_SCANCODE_G:
            gpu = !gpu;
            break;

          case SDL_SCANCODE_V:
            fieldlines = !fieldlines;
            break;

          case SDL_SCANCODE_COMMA:
            tick_period *= pow(2,0.25);
            break;

          case SDL_SCANCODE_PERIOD:
            tick_period /= pow(2,0.25);
            break;

          case SDL_SCANCODE_LEFTBRACKET:
            decay_rate *= pow(2,0.25);
            break;

          case SDL_SCANCODE_RIGHTBRACKET:
            decay_rate /= pow(2,0.25);
            break;

          default:
            break;
        }

      case SDL_WINDOWEVENT:
        if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
        {
          //cerr << event.window.data1 << " " << event.window.data2 << endl;
          reshape(event.window.data1, event.window.data2);
        }
        break;
    }
  }
  return false;
}

int main(int argc, char *argv[])
{
  //SDL Window/OpenGL Context
  SDL_Window* window = NULL;
  SDL_GLContext context;

  //Initialize
  if (init(&window, &context) != true)
  {
    cerr << "Shutting Down\n";
    return 1;
  }

  h_gvels = new float[4*M*M*M];
  //h_gtemp = new float[M*M*M];

  verts = new float[3*N];
  pvels = new float[3*N];
  times = new float[N];
  colors= new float[3*N];
  //memset(verts, 0.0, 3*N*sizeof(float));
  //memset(pvels  ,0.0, 3*N*sizeof(float));
  for (int i=0; i < 3*N; i += 3) {
    verts[i  ] = 8*((float)rand()/(float)RAND_MAX - 0.5) + M/2;
    verts[i+1] = 8*((float)rand()/(float)RAND_MAX - 0.5) + M/2;
    verts[i+2] = 8*((float)rand()/(float)RAND_MAX - 0.5) + M/2;
    pvels[i  ] = ((float)rand()/(float)RAND_MAX - 0.5)/1000.0;
    pvels[i+1] = ((float)rand()/(float)RAND_MAX - 0.5)/1000.0;
    pvels[i+2] = ((float)rand()/(float)RAND_MAX - 0.5)/1000.0;
    //verts[i] = 0;
    //pvels[i] = 0;
  }
  for (int i=0; i < N; ++i)
    //times[i]= ((float)rand()/(float)RAND_MAX);
    times[i]= (float)(i)/N;

  //allocate particle  and grid arrays
  if(cudaSuccess != cudaMalloc(&dverts, 3*N*sizeof(float))) cout << "failure to allocate\n";
  if(cudaSuccess != cudaMalloc(&dpvels, 3*N*sizeof(float))) cout << "failure to allocate\n";
  if(cudaSuccess != cudaMalloc(&dtimes,   N*sizeof(float))) cout << "failure to allocate\n";
  if(cudaSuccess != cudaMalloc(&dcolors,3*N*sizeof(float))) cout << "failure to allocate\n";

  if(cudaSuccess != cudaMalloc(&d_gvels[0],4*M*M*M*sizeof(float))) cout << "failure to allocate\n";
  if(cudaSuccess != cudaMalloc(&d_gvels[1],4*M*M*M*sizeof(float))) cout << "failure to allocate\n";
  //if(cudaSuccess != cudaMalloc(&d_gtemp[0],  M*M*M*sizeof(float))) cout << "failure to allocate\n";
  //if(cudaSuccess != cudaMalloc(&d_gtemp[1],  M*M*M*sizeof(float))) cout << "failure to allocate\n";
  //if(cudaSuccess != cudaMalloc(&d_gdens[0],  M*M*M*sizeof(float))) cout << "failure to allocate\n";
  //if(cudaSuccess != cudaMalloc(&d_gdens[1],  M*M*M*sizeof(float))) cout << "failure to allocate\n";
  //if(cudaSuccess != cudaMalloc(&d_gpres[0],  M*M*M*sizeof(float))) cout << "failure to allocate\n";
  //if(cudaSuccess != cudaMalloc(&d_gpres[1],  M*M*M*sizeof(float))) cout << "failure to allocate\n";
  //if(cudaSuccess != cudaMalloc(&d_diverge,   M*M*M*sizeof(float))) cout << "failure to allocate\n";

  //memset(zeros, 0.0, 4*M*M*M*sizeof(float));
  cudaError_t err;
  err = cudaMemcpy(d_gvels[0], zeros, 4*M*M*M*sizeof(float), cudaMemcpyHostToDevice); if (err) cout << "failure to memcpy: " << cudaGetErrorString(err) << endl;
  err = cudaMemcpy(d_gvels[1], zeros, 4*M*M*M*sizeof(float), cudaMemcpyHostToDevice); if (err) cout << "failure to memcpy: " << cudaGetErrorString(err) << endl;
  //err = cudaMemcpy(d_gtemp[0], zeros,   M*M*M*sizeof(float), cudaMemcpyHostToDevice); if (err) cout << "failure to memcpy: " << cudaGetErrorString(err) << endl;
  //err = cudaMemcpy(d_gtemp[1], zeros,   M*M*M*sizeof(float), cudaMemcpyHostToDevice); if (err) cout << "failure to memcpy: " << cudaGetErrorString(err) << endl;
  //err = cudaMemcpy(d_gpres[0], zeros,   M*M*M*sizeof(float), cudaMemcpyHostToDevice); if (err) cout << "failure to memcpy: " << cudaGetErrorString(err) << endl;
  //err = cudaMemcpy(d_gpres[1], zeros,   M*M*M*sizeof(float), cudaMemcpyHostToDevice); if (err) cout << "failure to memcpy: " << cudaGetErrorString(err) << endl;
  //err = cudaMemcpy(d_diverge,  zeros,   M*M*M*sizeof(float), cudaMemcpyHostToDevice); if (err) cout << "failure to memcpy: " << cudaGetErrorString(err) << endl;
  //
  //err = cudaMemcpy(d_gdens[0], zeros,   M*M*M*sizeof(float), cudaMemcpyHostToDevice); if (err) cout << "failure to memcpy: " << cudaGetErrorString(err) << endl;
  //err = cudaMemcpy(d_gdens[1], zeros,   M*M*M*sizeof(float), cudaMemcpyHostToDevice); if (err) cout << "failure to memcpy: " << cudaGetErrorString(err) << endl;
  if(cudaSuccess != cudaMemcpy(dverts, verts, 3*N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";
  if(cudaSuccess != cudaMemcpy(dtimes, times,   N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";
  if(cudaSuccess != cudaMemcpy(dcolors,colors,3*N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";
  if(cudaSuccess != cudaMemcpy(dpvels, pvels, 3*N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";

  //allocate Cuda random state
  if(cudaSuccess != cudaMalloc(&curandstate, 3*N*sizeof(curandState_t))) cout << "failure to allocate state\n";
  init_rand_state<<<3*N/512,512>>>(curandstate);

  if (err) quit = true;

  //////////////////////////////////////////////////////

  starTexture = LoadTexture("star.bmp");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  //Timing
  float r = 0;
  float dr = 0;
  float oldr = 0;
  //int Pause = 0;
  int frames = 0;

  //shader
  shader = CreateShaderProgGeom((char*)"flame.vert", (char*)"flame.geom", (char*)"flame.frag");
  pixlight = CreateShaderProg((char*)"pixlight.vert", (char*)"pixlight.frag");

  reshape(w,h);

  int startuptime = SDL_GetTicks();
  oldr = startuptime;

  ////////Main Loop////////
  //bool quit = false;
  try {
    while (!quit)
    {
      //cout << "handling events\n";
      quit = handleEvents();

      ////Physics Timing////
      r = SDL_GetTicks();
      dr += r - oldr;
      while (dr >= tick_period)
      {
        // 1000/8 = 125 updates per second
        physics(r);
        dr -= tick_period;
      }
      oldr = r;
      display(window, r);
      frames += 1;
      //quit = true;
    }
  }
  catch (...) {cout << "catch block\n";}

  cout << "Shutting Down\n";
  cout << "average framerate: " << 1000*(float)frames/(r - startuptime) << endl;

  cudaFree(dverts);
  cudaFree(dpvels);
  cudaFree(dtimes);
  cudaFree(dcolors);
  cudaFree(d_gvels[0]);
  cudaFree(d_gvels[1]);
  cudaFree(curandstate);
  //cudaFree(d_gtemp[0]);
  //cudaFree(d_gtemp[1]);
  //cudaFree(d_gdens[0]);
  //cudaFree(d_gdens[1]);
  //cudaFree(d_gpres[0]);
  //cudaFree(d_gpres[1]);
  //cudaFree(d_diverge);
  delete verts;
  delete pvels;
  delete times;
  delete colors;

  delete h_gvels;
  //delete h_gtemp;

  SDL_Quit();

  return 0;
}
