#include <stdio.h>
#include "timer.h"
#include <math.h>
#define PG_SIZE (4096)
#define BYTE_SIZE (256)
#define BLOCK_SIZE 32

//각 block이 공유하는 shared memory


__global__
void similarity(int data_size, char *data_old, char *data_new, int* sim)
{
  //dynamic shared memory
  int gthread_idx = blockIdx.x*blockDim.x + threadIdx.x;
 // int lthread_idx = threadIdx.x;
  int size = PG_SIZE; //각 쓰레드가 담당할 데이터 영역 크기
  int gbuf_idx = gthread_idx * size;
  int i;
  //init shared variable
  int sim_cnt=0;

  for(i=gbuf_idx; i<gbuf_idx+size && i<data_size; i++)
  {
    //similarity calculation
    sim_cnt += (data_old[i] == data_new[i]);
   }
   sim[gthread_idx] = sim_cnt;
   //printf("%d %d\n", lthread_idx, sim_cnt);
}

int main(void)
{
  int data_size = 1<<30;//1<<30;  //1GB
  char *data_old, *data_new;
  char *d_do, *d_dn;
  int num_threads;
  int num_blocks;
  int *sim, *d_sim;
  double start, finish;
  //int i;
  //데이터 초기화
  data_old = (char*)malloc(data_size*sizeof(char)); //1GB
  data_new = (char*)malloc(data_size*sizeof(char)); //1GB
  sim = (int*)malloc((data_size/PG_SIZE)*sizeof(int)); //256K*12B = 3KB

  //디바이스 데이터 할당
  cudaMalloc(&d_do, data_size*sizeof(char));  //1GB
  cudaMalloc(&d_dn, data_size*sizeof(char));  //1GB
  cudaMalloc(&d_sim, (data_size/PG_SIZE) * sizeof(int)); //3KB

  //데이터 초기화
  for (int i = 0; i < data_size; i++) {
    data_old[i] = 1;
    data_new[i] = 1;
  }

  printf("Start Evaluation\n");
  printf("1. memcpy\n");

  GET_TIME(start);
  cudaMemcpy(d_do, data_old, data_size*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dn, data_new, data_size*sizeof(char), cudaMemcpyHostToDevice);

  //block size 설정 (조정가능)
  num_threads = 32;
  num_blocks = data_size/PG_SIZE/num_threads; //1GB/32?? 상관무
  printf("nthread, nblock: %d %d\n", num_threads, num_blocks);
  // Perform sim_ent function on 1GB elements
  //3번째인자는 shared memory의 크기 = 516Bytes * 32 = 16KB (L1cache size 48KB)
  similarity<<<num_blocks, num_threads>>>(data_size, d_do, d_dn, d_sim);

  cudaMemcpy(sim, d_sim, (data_size/PG_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);
  
 // for(int i=0; i<data_size/PG_SIZE; i++)
 //   printf("sim: %d\n", sim[i]);
  GET_TIME(finish);

  printf("Elapsed time = %e seconds\n", finish - start);

  cudaFree(d_do);
  cudaFree(d_dn);
  cudaFree(d_sim);
  
  free(data_old);
  free(data_new);
  free(sim);
}