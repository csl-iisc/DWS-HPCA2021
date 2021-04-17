/*
   this microbenchmark is for testing TLB variations
 */

#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cutil_inline.h>
#include <unistd.h>
#include "benchmark_common.h"
#include <iomanip>
#include <vector>

#include "../src/gpgpu-sim/ConfigOptions.h"
#include "../src/gpgpu-sim/App.h"
#include "../src/common.h"

thread_to_appID_struct* thread_to_appID;

struct app_data {
  app_data(char* app_name, pthread_mutex_t* app_mutex, bool concurrent, cudaEvent_t* done,
      std::vector<cudaEvent_t>* done_events, size_t app_num) :
    app_name(app_name), app_mutex(app_mutex), concurrent(concurrent), done(done),
    done_events(done_events), appID(app_num) {
      cutilSafeCall(cudaStreamCreate(&stream));
    }
  cudaStream_t stream;
  cudaEvent_t* done;
  std::vector<cudaEvent_t>* done_events;
  char* app_name;
  pthread_mutex_t* app_mutex;
  bool concurrent;
  size_t appID;
};

#define N 32
#define ITER 1
#define JUMP 7
#define BLOCKS 30
#define THREADS 32
#define STRIDE 1024

__global__
void add(int *a, int *b) {
  int bi = blockIdx.x;
  int ti = threadIdx.x;
  int eti = bi * THREADS + ti;
  int sum = 0;
  /* if(ti % 32 == 0) */
  /* { */
  for(int i = 0; i < ITER; i++)
  {
    for(int j = 0; j < JUMP; j++)
    {
      int index = STRIDE * ( JUMP * eti + j);
      /* int index = STRIDE * ( JUMP * ti + j); */
      /* int index = j * eti; */
      /* int index = 4096 * ( JUMP * ti + j); */
      /* a[index] = sum; */
      //
      int offset = (ti % 32) * 32;
      /* int offset = 0; */
      sum += a[index + offset];
    }
  }
  int index = STRIDE * ( JUMP * eti);
  /* int index = STRIDE * ( JUMP * ti); */
  int offset = (ti % 32) * 32;
      /* int offset = 0; */
  a[index + offset] = sum;
  /* } */
}

int main() {
  int n_apps = 1;
  std::vector<app_data> apps;
  std::vector<cudaEvent_t> done_events(n_apps, NULL);

  pthread_mutex_t app_mutex;
  pthread_mutex_init(&app_mutex, NULL);

  thread_to_appID = (thread_to_appID_struct *)malloc(sizeof(thread_to_appID_struct));
  thread_to_appID->init();

  ConfigOptions::n_apps = n_apps;

  cutilSafeCall(cudaEventCreate(&done_events[0]));
  apps.push_back(app_data("simple", &app_mutex, false, &done_events[0], &done_events, 0));

  struct app_data *app = &apps[0];
  printf("---------------------------%d stream\n\n\n", app->stream);

  printf("Launch code in main.cu:launching a new benchmark, appID = %d, already registered? = %d\n", app->appID, App::is_registered(app->appID));

  if(App::is_registered(app->appID)) thread_to_appID->add((void*)pthread_self(), App::get_app_id(app->appID));
  else thread_to_appID->add((void*)pthread_self(), App::register_app(app->appID));


  /* 4 gb */
  int size = STRIDE * JUMP * BLOCKS * THREADS;
  /* int size = 32 * 4096; */
  int *ha, *hb;
  ha = (int*) malloc(sizeof(int) * size);
  hb = (int*) malloc(sizeof(int) * size);
  if(ha == NULL || hb == NULL)
  {
    printf("memory allocation failed\n");
    exit(0);
  }

  int *da, *db;
  cudaMalloc((void **)&da, size*sizeof(int));
  cudaMalloc((void **)&db, size*sizeof(int));

  /* for (int i = 0; i<N; i+=32) { */
  /*   ha[i] = i; */
  /* } */

  cudaMemcpyAsync(da, ha, size*sizeof(int), cudaMemcpyHostToDevice, app->stream);

    add<<<BLOCKS, THREADS, 0, app->stream>>>(da, db);
    /* add<<<BLOCKS, THREADS, 0, app->stream>>>(da, db); */

    /* cutilSafeCall(cudaThreadSynchronize()); */
    cutilSafeCall(cudaStreamSynchronize(app->stream));
    /* add<<<1, N, 0, app->stream>>>(da, db); */
    /* cutilSafeCall(cudaStreamSynchronize(app->stream)); */

  cudaMemcpyAsync(hb, db, size*sizeof(int), cudaMemcpyDeviceToHost, app->stream);

  cutilSafeCall(cudaEventRecord(*app->done, app->stream));

  /* cutilSafeCall(cudaStreamSynchronize(app->stream)); */
  /* /1* cutilSafeCall(cudaThreadSynchronize()); *1/ */

  /* cudaFree(da); */
  /* cudaFree(db); */


  bool some_running;

  do
  {
    some_running = false;
    for (std::vector<cudaEvent_t>::iterator e = app->done_events->begin();
        e != app->done_events->end(); e++) {
      if (cudaEventQuery(*e) == cudaErrorNotReady) {
        some_running = true;
        break;
      }
    }
    sleep(1);
  }while(some_running);
  sleep(1);


  cutilSafeCall(cudaStreamDestroy(apps[0].stream));

  /* for (int i = 0; i<N; ++i) { */
  /*   printf("%d\n", hb[i]); */
  /* } */

  return 0;
}

