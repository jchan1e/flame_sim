#include "stdGL.h"
#include <vector>
#include <iostream>
#include "objects.h"
#include "shader.h"
#include "helper_math.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_image.h>

using namespace std;

//GLOBAL VARIABLES//
//running or not
//bool quit = false;

int Pause = 0;

//Window Size
int w = 1920;
int h = 1080;

//eye position and orientation
double ex = 0;
double ey = 0;
double ez = 0;
double zoom = 10;
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

// Array Sizes
//const int N = pow(2,13);
const int N = pow(2,13);
const int M = 64;
int ping = 0;
int pong = 1;

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
float* gvels  = NULL;
float* gtemp  = NULL;
float* gdens  = NULL;

cudaSurfaceObject_t s_gvels0;
cudaSurfaceObject_t s_gtemp0;
cudaSurfaceObject_t s_gdens0;
cudaSurfaceObject_t s_gpres0;
cudaSurfaceObject_t s_gvels1;
cudaSurfaceObject_t s_gtemp1;
cudaSurfaceObject_t s_gdens1;
cudaSurfaceObject_t s_gpres1;
cudaTextureObject_t t_gvels[2] = {0};
cudaTextureObject_t t_gtemp[2] = {0};
cudaTextureObject_t t_gdens[2] = {0};
cudaTextureObject_t t_gpres[2] = {0};

//User-controlled Computation Modes
bool stepmode = true;
bool gpu = true;

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

__global__ void pstep(cudaTextureObject_t gvels, cudaTextureObject_t gtemp, float* verts, float* times, float* colors) {
  int I = blockIdx.x*blockDim.x + threadIdx.x;
  int i = I * 3;
  float4 V = tex3D<float4>(gvels, verts[i], verts[i+1], verts[+2]);
  //verts[i  ] += V.x;
  //verts[i+1] += V.y;
  //verts[i+2] += V.z;
  verts[i  ] = 0.0;
  verts[i+1] = 0.0;
  verts[i+2] = 0.0;

  times[I] -= 0.0001f;
  colors[i  ] = sqrt(times[I]);
  colors[i+1] = max(times[I]/1.125f, 0.0f);
  colors[i+2] = pow(times[I],2.0f);

  //if (times[I] > 0.0) {
  //  float tmp = tex3D<float4>(gtemp, verts[i], verts[i+1], verts[i+2]).x;
  //  surf3Dwrite(tmp+times[I]*0.01, s_gtemp0, __float2int_rz(verts[i]), __float2int_rz(verts[i+1]), __float2int_rz(verts[i+2]), cudaBoundaryModeTrap);
  //}

  // move this to the CPU and initialize position randomly within a sphere
  if (times[I] <= -0.5f) {
    times[I]   = 1.0f;
    verts[i  ] = V.x;
    verts[i+1] = V.y;
    verts[i+2] = V.z;
  }
}

__device__ void advect(cudaTextureObject_t vels, cudaTextureObject_t src, surface<void,cudaSurfaceType3D> dest, float dissipation,
                       int I, int J, int K) {

  float dt = 0.016;

  float4 v = tex3D<float4>(vels, I, J, K);
  float i = dt*v.x;
  float j = dt*v.y;
  float k = dt*v.z;
  float4 V = dissipation * tex3D<float4>(src, i, j, k);
  surf3Dwrite(V, dest, I, J, K, cudaBoundaryModeTrap);
}

__device__ void buoyancy(cudaTextureObject_t gvels, cudaTextureObject_t gtemp, cudaTextureObject_t gdens, surface<void,cudaSurfaceType3D> dest, int i, int j, int k) {
  float ambient = 0.0;
  float dt = 0.016;
  float sigma = 1.0;
  float kappa = 0.05;

  float T = tex3D<float4>(gtemp, i, j, k).x;
  float4 V = tex3D<float4>(gvels, i, j, k);
  float4 R = V;
  float D = tex3D<float4>(gdens, i, j, k).x;
  R.y += dt * (T - ambient) * sigma - D * kappa;
  surf3Dwrite(R, dest, i, j, k, cudaBoundaryModeTrap);
}

__device__ float divergence(cudaTextureObject_t gvels, int i, int j, int k) {
  float vr = tex3D<float4>(gvels, i+1, j  , k  ).x;
  float vl = tex3D<float4>(gvels, i-1, j  , k  ).x;
  float vf = tex3D<float4>(gvels, i  , j+1, k  ).y;
  float vb = tex3D<float4>(gvels, i  , j-1, k  ).y;
  float vu = tex3D<float4>(gvels, i  , j  , k+1).z;
  float vd = tex3D<float4>(gvels, i  , j  , k-1).z;
  return vr-vl + vf-vb + vu-vd;
}

__device__ void jacobi(cudaTextureObject_t gpres0, float diverg, surface<void,cudaSurfaceType3D> gpres1, int i, int j, int k) {
  float alpha = 1.25;
  float beta = 1.0/6.0;
  float pr = tex3D<float4>(gpres0, i+1, j  , k  ).x;
  float pl = tex3D<float4>(gpres0, i-1, j  , k  ).x;
  float pf = tex3D<float4>(gpres0, i  , j+1, k  ).x;
  float pb = tex3D<float4>(gpres0, i  , j-1, k  ).x;
  float pu = tex3D<float4>(gpres0, i  , j  , k+1).x;
  float pd = tex3D<float4>(gpres0, i  , j  , k-1).x;
  float R = (pr+pl+pf+pb+pu+pd + alpha*diverg) * beta;
  surf3Dwrite(R, gpres1, i, j, k, cudaBoundaryModeTrap);
}

__device__ void gradient(cudaTextureObject_t gvel0, cudaTextureObject_t gpres, surface<void,cudaSurfaceType3D> gvel1, int i, int j, int k) {
  float pr = tex3D<float4>(gpres, i+1, j  , k  ).x;
  float pl = tex3D<float4>(gpres, i-1, j  , k  ).x;
  float pf = tex3D<float4>(gpres, i  , j+1, k  ).x;
  float pb = tex3D<float4>(gpres, i  , j-1, k  ).x;
  float pu = tex3D<float4>(gpres, i  , j  , k+1).x;
  float pd = tex3D<float4>(gpres, i  , j  , k-1).x;
  float gradscale = 0.9;
  float4 oldV = tex3D<float4>(gvel0, i, j, k);
  float4 grad = make_float4(pr-pl, pf-pb, pu-pd, 0.0)*gradscale;
  float4 newV = oldV-grad;
  surf3Dwrite(newV, gvel1, i, j, k, cudaBoundaryModeTrap);
}

__global__ void gstep(cudaTextureObject_t gvelA, cudaTextureObject_t gvelB, cudaTextureObject_t gtemp, cudaTextureObject_t gdens, cudaTextureObject_t gpres0, cudaTextureObject_t gpres1, float* verts, int ping, int pong) {

  int I = blockIdx.x*blockDim.x + threadIdx.x;
  int J = blockIdx.y*blockDim.y + threadIdx.y;
  int K = blockIdx.z*blockDim.z + threadIdx.z;

  surf3Dwrite(0.0f, gpres0, I, J, K, cudaBoundaryModeTrap);
  surf3Dwrite(0.0f, gpres1, I, J, K, cudaBoundaryModeTrap);

  cudaTextureObject_t gvel0;
  cudaTextureObject_t gvel1;
  if (pong) {
    gvel0 = gvelB;
    gvel1 = gvelA;
  }
  else {
    gvel0 = gvelA;
    gvel1 = gvelB;
  }

  advect(gvel0, gvel0, s_gvels1, 0.99  ,  I, J, K);
  advect(gvel0, gtemp, s_gtemp1, 0.99  ,  I, J, K);
  advect(gvel0, gdens, s_gdens1, 0.9999,  I, J, K);
  buoyancy(gvel1, gtemp, gdens, s_gvels0, I, J, K);
  float D = divergence(gvel0, I, J, K);
  for (int i=0; i < 40/2; ++i) {
    jacobi(gpres0, D, s_gpres1, I, J, K);
    jacobi(gpres1, D, s_gpres0, I, J, K);
  }
  gradient(gvel0, gpres0, s_gvels1, I, J, K);
}

void step_gpu(float* verts, float* times, float* colors,
              cudaTextureObject_t gvel0, cudaTextureObject_t gvel1, cudaTextureObject_t gtemp0, cudaTextureObject_t gtemp1, cudaTextureObject_t gdens, cudaTextureObject_t gpres0, cudaTextureObject_t gpres1,
              //surface<void,cudaSurfaceType3D> s_gvels, surface<void,cudaSurfaceType3D> s_gtemp, surface<void,cudaSurfaceType3D> s_gdens,
              const int N, const int M, int ping, int pong) {
  dim3 gblock(M/8,M/8,M/8);
  dim3 gthread(8,8,8);
  gstep<<<gblock, gthread>>>(gvel0, gvel1, gtemp0, gdens, gpres0, gpres1, verts, ping, pong);
  if (cudaSuccess != cudaGetLastError()) ;//cout << "gstep error\n";
  cudaTextureObject_t gvel;
  if (pong) gvel = gvel0;
  else      gvel = gvel1;
  int pblock = N/32;
  int pthread = 32;
  pstep<<<pblock, pthread>>>(gvel, gtemp1, verts, times, colors);
  if (cudaSuccess != cudaGetLastError()) ;//cout << "pstep error\n";
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
//////// SDL Init Function ////////

bool init(SDL_Window** window, SDL_GLContext* context)
{
  bool success = true;

  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
  {
    cerr << "SDL failed to initialize: " << SDL_GetError() << endl;
    success = false;
  }

  *window = SDL_CreateWindow("Flame", 0,0, 1920,1080, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
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
  //SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);


  //Vsync
  if (SDL_GL_SetSwapInterval(0) < 0)
  {
    cerr << "SDL could not set Vsync: " << SDL_GetError() << endl;
//    success = false;
  }

  //TTF_Font handling
  //if (TTF_Init() < 0)
  //{
  //  cerr << "TTF font library could not be initialized: " << TTF_GetError() << endl;
  //  success = false;
  //}

  cout << SDL_GetError() << endl;
  return success;
}

///////////////////////////////////

void display(SDL_Window* window)
{
  //// Step Flame Animation ////
  if (stepmode) {
    if (gpu) {
      if(cudaSuccess != cudaMemcpy(dverts, verts, 3*N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";
      if(cudaSuccess != cudaMemcpy(dtimes, times,   N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";
      if(cudaSuccess != cudaMemcpy(dcolors,colors,3*N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";
      step_gpu(dverts, dtimes, dcolors,
               t_gvels[0], t_gvels[1], t_gtemp[0], t_gtemp[1], t_gdens[ping], t_gpres[0], t_gpres[1],
               //s_gvels[pong], s_gtemp[pong], s_gdens[pong],
               N, M, ping, pong);
      ping = pong;
      pong = 1-pong;
      if(cudaSuccess != cudaMemcpy(verts, dverts, 3*N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail\n";
      if(cudaSuccess != cudaMemcpy(times, dtimes,   N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail\n";
      if(cudaSuccess != cudaMemcpy(colors,dcolors,3*N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail\n";
    }
    else {
      step_cpu(verts, pvels, times, colors, N);
    }
  }
  //////////////////////////////

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

  gluLookAt(ex,ey,ez+5.0, 0,0,5.0, 0,0,Cos(ph));

  // lighting
  glEnable(GL_LIGHTING);
  float white[4]   = {1.0,1.0,1.0,1.0};
  float pos[4]     = {0.0, 0.0, 8.0, 1.0};
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

  glUseProgram(pixlight);
  glColor3f(1.0,1.0,1.0);
  ball(0,0,0, 0.5);

  glUseProgram(shader);
  glDisable(GL_DEPTH_TEST);
  glBindTexture(GL_TEXTURE_2D, starTexture);
  int id = glGetUniformLocation(shader, "star");
  if (id>=0) glUniform1i(id,0);
  // ^ current bound texture, star.bmp
  id = glGetUniformLocation(shader, "size");
  if (id>=0) glUniform1f(id,0.4);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE,GL_ONE);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glVertexPointer(3,GL_FLOAT,0,verts);
  glColorPointer(3,GL_FLOAT,0,colors);

  glDrawArrays(GL_POINTS,0,N);

  glDisable(GL_BLEND);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  //swap the buffers
  glFlush();
  SDL_GL_SwapWindow(window);
}

void physics()
{
  const Uint8* state = SDL_GetKeyboardState(NULL);
  keyboard(state);

  //adjust the eye position
  th += dth;
  ph += dph;
  zoom = zoom<2.0?2.0:zoom+dzoom;

  //// Step Flame Animation ////
  if (!stepmode) {
    if (gpu) {
      if(cudaSuccess != cudaMemcpy(dverts, verts, 3*N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";
      if(cudaSuccess != cudaMemcpy(dtimes, times,   N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";
      if(cudaSuccess != cudaMemcpy(dcolors,colors,3*N*sizeof(float), cudaMemcpyHostToDevice)) cout << "memcpy fail\n";
      step_gpu(dverts, dtimes, dcolors,
               t_gvels[0], t_gvels[1], t_gtemp[0], t_gtemp[1], t_gdens[ping], t_gpres[0], t_gpres[1],
               //s_gvels[pong], s_gtemp[pong], s_gdens[pong],
               N, M, ping, pong);
      ping = pong;
      pong = 1-pong;
      if(cudaSuccess != cudaMemcpy(verts, dverts, 3*N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail\n";
      if(cudaSuccess != cudaMemcpy(times, dtimes,   N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail\n";
      if(cudaSuccess != cudaMemcpy(colors,dcolors,3*N*sizeof(float), cudaMemcpyDeviceToHost)) cout << "memcpy fail\n";
    }
    else {
      step_cpu(verts, pvels, times, colors, N);
    }
  }
  //////////////////////////////
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
  gluPerspective(60, w2h, 1.0, 2*M);

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

  verts = new float[3*N];
  pvels = new float[3*N];
  times = new float[N];
  colors= new float[3*N];
  //memset(verts, 0.0, 3*N*sizeof(float));
  //memset(pvels  ,0.0, 3*N*sizeof(float));
  for (int i=0; i < 3*N; ++i) {
    pvels[i] = ((float)rand()/(float)RAND_MAX - 0.5)/1000.0;
  }
  for (int i=0; i < N; ++i)
    times[i]= ((float)rand()/(float)RAND_MAX);

  //allocate particle arrays
  if(cudaSuccess != cudaMalloc(&dverts, 3*N*sizeof(float))) cout << "failure to allocate\n";;
  if(cudaSuccess != cudaMalloc(&dpvels, 3*N*sizeof(float))) cout << "failure to allocate\n";;
  if(cudaSuccess != cudaMalloc(&dtimes,   N*sizeof(float))) cout << "failure to allocate\n";;
  if(cudaSuccess != cudaMalloc(&dcolors,3*N*sizeof(float))) cout << "failure to allocate\n";;

  //////////////////////////////////////////////////////
  //set up grid textures, surfaces, and arrays in cuda
	struct cudaTextureDesc texDesc[8];
  //memset(&texDesc[0], 0, sizeof(texDesc));
  texDesc[0].addressMode[0]   = cudaAddressModeWrap;
  texDesc[0].addressMode[1]   = cudaAddressModeWrap;
  texDesc[0].addressMode[2]   = cudaAddressModeClamp;
  texDesc[0].filterMode       = cudaFilterModeLinear;
  texDesc[0].readMode         = cudaReadModeElementType;
  texDesc[0].normalizedCoords = 0;

  struct cudaResourceDesc r_gvels0;
  //memset(&r_gvels0, 0, sizeof(r_gvels0));
  r_gvels0.resType = cudaResourceTypeArray;
  cudaChannelFormatDesc ch_gvels0 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat );
  cudaArray *d_gvels0;
  if(cudaSuccess != cudaMalloc3DArray(&d_gvels0, &ch_gvels0, make_cudaExtent(M,M,M), cudaArraySurfaceLoadStore)) cout << "failure to allocate tex1\n";
  r_gvels0.res.array.array = d_gvels0;
  
  if(cudaSuccess != cudaCreateTextureObject(&t_gvels[0], &r_gvels0, &texDesc[0], NULL)) cout << "failure to create texture object\n";
  if(cudaSuccess != cudaCreateSurfaceObject(&s_gvels0, &r_gvels0)) cout << "failure to bind surface\n";

  //memset(&texDesc[1], 0, sizeof(texDesc));
  texDesc[1].addressMode[0]   = cudaAddressModeWrap;
  texDesc[1].addressMode[1]   = cudaAddressModeWrap;
  texDesc[1].addressMode[2]   = cudaAddressModeClamp;
  texDesc[1].filterMode       = cudaFilterModeLinear;
  texDesc[1].readMode         = cudaReadModeElementType;
  texDesc[1].normalizedCoords = 0;

  struct cudaResourceDesc r_gvels1;
  //memset(&r_gvels1, 0, sizeof(r_gvels1));
  r_gvels1.resType = cudaResourceTypeArray;
  cudaChannelFormatDesc ch_gvels1 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat );
  cudaArray *d_gvels1;
  if(cudaSuccess != cudaMalloc3DArray(&d_gvels1, &ch_gvels1, make_cudaExtent(M,M,M), cudaArraySurfaceLoadStore)) cout << "failure to allocate tex2\n";
  r_gvels1.res.array.array = d_gvels1;
  
  if(cudaSuccess != cudaCreateTextureObject(&t_gvels[1], &r_gvels1, &texDesc[1], NULL)) cout << "failure to create texture object\n";
  if(cudaSuccess != cudaCreateSurfaceObject(&s_gvels1, &r_gvels1)) cout << "failure to bind surface\n";

  //memset(&texDesc[2], 0, sizeof(texDesc));
  texDesc[2].addressMode[0]   = cudaAddressModeWrap;
  texDesc[2].addressMode[1]   = cudaAddressModeWrap;
  texDesc[2].addressMode[2]   = cudaAddressModeClamp;
  texDesc[2].filterMode       = cudaFilterModeLinear;
  texDesc[2].readMode         = cudaReadModeElementType;
  texDesc[2].normalizedCoords = 0;

  struct cudaResourceDesc r_gtemp0;
  //memset(&r_gtemp0, 0, sizeof(r_gtemp0));
  r_gtemp0.resType = cudaResourceTypeArray;
  cudaChannelFormatDesc ch_gtemp0 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat );
  cudaArray *d_gtemp0;
  if(cudaSuccess != cudaMalloc3DArray(&d_gtemp0, &ch_gtemp0, make_cudaExtent(M,M,M), cudaArraySurfaceLoadStore)) cout << "failure to allocate tex3\n";
  r_gtemp0.res.array.array = d_gtemp0;
  
  cudaCreateTextureObject(&t_gtemp[0], &r_gtemp0, &texDesc[2], NULL);
  cudaCreateSurfaceObject(&s_gtemp0, &r_gtemp0);

  //memset(&texDesc[3], 0, sizeof(texDesc));
  texDesc[3].addressMode[0]   = cudaAddressModeWrap;
  texDesc[3].addressMode[1]   = cudaAddressModeWrap;
  texDesc[3].addressMode[2]   = cudaAddressModeClamp;
  texDesc[3].filterMode       = cudaFilterModeLinear;
  texDesc[3].readMode         = cudaReadModeElementType;
  texDesc[3].normalizedCoords = 0;

  struct cudaResourceDesc r_gtemp1;
  //memset(&r_gtemp1, 0, sizeof(r_gtemp1));
  r_gtemp0.resType = cudaResourceTypeArray;
  cudaChannelFormatDesc ch_gtemp1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat );
  cudaArray *d_gtemp1;
  if(cudaSuccess != cudaMalloc3DArray(&d_gtemp1, &ch_gtemp1, make_cudaExtent(M,M,M), cudaArraySurfaceLoadStore)) cout << "failure to allocate tex4\n";
  r_gtemp0.res.array.array = d_gtemp1;
  
  cudaCreateTextureObject(&t_gtemp[1], &r_gtemp1, &texDesc[3], NULL);
  cudaCreateSurfaceObject(&s_gtemp1, &r_gtemp1);

  //memset(&texDesc[4], 0, sizeof(texDesc));
  texDesc[4].addressMode[0]   = cudaAddressModeWrap;
  texDesc[4].addressMode[1]   = cudaAddressModeWrap;
  texDesc[4].addressMode[2]   = cudaAddressModeClamp;
  texDesc[4].filterMode       = cudaFilterModeLinear;
  texDesc[4].readMode         = cudaReadModeElementType;
  texDesc[4].normalizedCoords = 0;

  struct cudaResourceDesc r_gdens0;
  //memset(&r_gdens0, 0, sizeof(r_gdens0));
  r_gdens0.resType = cudaResourceTypeArray;
  cudaChannelFormatDesc ch_gdens0 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat );
  cudaArray *d_gdens0;
  if(cudaSuccess != cudaMalloc3DArray(&d_gdens0, &ch_gdens0, make_cudaExtent(M,M,M), cudaArraySurfaceLoadStore)) cout << "failure to allocate tex5\n";
  r_gdens0.res.array.array = d_gdens0;
  
  cudaCreateTextureObject(&t_gdens[0], &r_gdens0, &texDesc[4], NULL);
  cudaCreateSurfaceObject(&s_gdens0, &r_gdens0);

  //memset(&texDesc[5], 0, sizeof(texDesc));
  texDesc[5].addressMode[0]   = cudaAddressModeWrap;
  texDesc[5].addressMode[1]   = cudaAddressModeWrap;
  texDesc[5].addressMode[2]   = cudaAddressModeClamp;
  texDesc[5].filterMode       = cudaFilterModeLinear;
  texDesc[5].readMode         = cudaReadModeElementType;
  texDesc[5].normalizedCoords = 0;

  struct cudaResourceDesc r_gdens1;
  //memset(&r_gdens1, 0, sizeof(r_gdens1));
  r_gdens0.resType = cudaResourceTypeArray;
  cudaChannelFormatDesc ch_gdens1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat );
  cudaArray *d_gdens1;
  if(cudaSuccess != cudaMalloc3DArray(&d_gdens1, &ch_gdens1, make_cudaExtent(M,M,M), cudaArraySurfaceLoadStore)) cout << "failure to allocate tex6\n";
  r_gdens0.res.array.array = d_gdens1;
  
  cudaCreateTextureObject(&t_gdens[1], &r_gdens1, &texDesc[5], NULL);
  cudaCreateSurfaceObject(&s_gdens1, &r_gdens1);

  //memset(&texDesc[5], 0, sizeof(texDesc));
  texDesc[6].addressMode[0]   = cudaAddressModeWrap;
  texDesc[6].addressMode[1]   = cudaAddressModeWrap;
  texDesc[6].addressMode[2]   = cudaAddressModeClamp;
  texDesc[6].filterMode       = cudaFilterModeLinear;
  texDesc[6].readMode         = cudaReadModeElementType;
  texDesc[6].normalizedCoords = 0;

  struct cudaResourceDesc r_gpres0;
  //memset(&r_gpres0, 0, sizeof(r_gpres0));
  r_gpres0.resType = cudaResourceTypeArray;
  cudaChannelFormatDesc ch_gpres0 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat );
  cudaArray *d_gpres0;
  if(cudaSuccess != cudaMalloc3DArray(&d_gpres0, &ch_gpres0, make_cudaExtent(M,M,M), cudaArraySurfaceLoadStore)) cout << "failure to allocate tex7\n";
  r_gpres0.res.array.array = d_gpres0;
  
  cudaCreateTextureObject(&t_gpres[0], &r_gpres0, &texDesc[6], NULL);
  cudaCreateSurfaceObject(&s_gpres0, &r_gpres0);

  //memset(&texDesc[5], 0, sizeof(texDesc));
  texDesc[7].addressMode[0]   = cudaAddressModeWrap;
  texDesc[7].addressMode[1]   = cudaAddressModeWrap;
  texDesc[7].addressMode[2]   = cudaAddressModeClamp;
  texDesc[7].filterMode       = cudaFilterModeLinear;
  texDesc[7].readMode         = cudaReadModeElementType;
  texDesc[7].normalizedCoords = 0;

  struct cudaResourceDesc r_gpres1;
  //memset(&r_gpres1, 0, sizeof(r_gpres1));
  r_gpres1.resType = cudaResourceTypeArray;
  cudaChannelFormatDesc ch_gpres1 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat );
  cudaArray *d_gpres1;
  if(cudaSuccess != cudaMalloc3DArray(&d_gpres1, &ch_gpres1, make_cudaExtent(M,M,M), cudaArraySurfaceLoadStore)) cout << "failure to allocate tex8\n";
  r_gpres1.res.array.array = d_gpres1;
  
  cudaCreateTextureObject(&t_gpres[1], &r_gpres1, &texDesc[7], NULL);
  cudaCreateSurfaceObject(&s_gpres1, &r_gpres1);

  //////////////////////////////////////////////////////

  starTexture = LoadTexture("star.bmp");
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  //Timing
  int r = 0;
  int dr = 0;
  int oldr = 0;
  //int Pause = 0;
  int frames = 0;

  //shader
  shader = CreateShaderProgGeom((char*)"flame.vert", (char*)"flame.geom", (char*)"flame.frag");
  pixlight = CreateShaderProg((char*)"pixlight.vert", (char*)"pixlight.frag");

  reshape(w,h);

  int startuptime = SDL_GetTicks();
  oldr = startuptime;

  ////////Main Loop////////
  bool quit = false;
  while (!quit)
  {
    //cout << "handling events\n";
    quit = handleEvents();

    ////Physics Timing////
    r = SDL_GetTicks();
    dr += r - oldr;
    while (dr >= 8)
    {
      // 1000/8 = 125 updates per second
      physics();
      dr -= 8;
    }
    oldr = r;
    display(window);
    frames += 1;
    //quit = true;
  }

  cout << "Shutting Down\n";
  cout << "average framerate: " << 1000*(float)frames/(r - startuptime) << endl;

  cudaFree(dverts);
  cudaFree(dpvels);
  cudaFree(dtimes);
  cudaFree(dcolors);
  cudaFreeArray(d_gvels0);
  cudaFreeArray(d_gvels1);
  cudaFreeArray(d_gtemp0);
  cudaFreeArray(d_gtemp1);
  cudaFreeArray(d_gdens0);
  cudaFreeArray(d_gdens1);
  cudaFreeArray(d_gpres0);
  cudaFreeArray(d_gpres1);
  cudaDestroyTextureObject(t_gvels[0]);
  cudaDestroyTextureObject(t_gvels[1]);
  cudaDestroyTextureObject(t_gtemp[0]);
  cudaDestroyTextureObject(t_gtemp[1]);
  cudaDestroyTextureObject(t_gdens[0]);
  cudaDestroyTextureObject(t_gdens[1]);
  cudaDestroyTextureObject(t_gpres[0]);
  cudaDestroyTextureObject(t_gpres[1]);
  delete verts;
  delete pvels;
  delete times;
  delete colors;

  SDL_Quit();

  return 0;
}
