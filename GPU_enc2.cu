#include <stdio.h>
#include "timer.h"
#include <math.h>
#define PG_SIZE (4096)
#define BYTE_SIZE (256)
#define BLOCK_SIZE 32

//각 block이 공유하는 shared memory
typedef struct{
  unsigned short int ent_cnt[BYTE_SIZE];
  unsigned short int sim_cnt;
}BLK_DET;

typedef struct{
  double ent_val;
  int sim_val;
}SIM_ENT;

__global__
void sim_ent(int data_size, char *data_old, char *data_new, SIM_ENT* res)
{
  //dynamic shared memory
  extern __shared__ BLK_DET detect[];
  int gthread_idx = blockIdx.x*blockDim.x + threadIdx.x;
  int lthread_idx = threadIdx.x;
  int size = PG_SIZE/blockDim.x; //각 쓰레드가 담당할 데이터 영역 크기
  int gbuf_idx = gthread_idx * size;
  int i, j;
  double Pb;
  //init shared variable
  detect[lthread_idx].sim_cnt=0;
  for(i=0; i<BYTE_SIZE; i++)
    detect[lthread_idx].ent_cnt[i]=0;

  for(i=gbuf_idx; i<gbuf_idx+size && i<data_size; i++)
  {
    //similarity calculation
    detect[lthread_idx].sim_cnt += (data_old[i] == data_new[i]);
    //entropy cacluation
    detect[lthread_idx].ent_cnt[data_new[i]]++;
  }
  __syncthreads();
  //reduction and summation
  for(i=1; i<blockDim.x; i+=i)
  {
    if((lthread_idx & ((1<<i)-1)) == 0)
    {
      detect[lthread_idx].sim_cnt += detect[lthread_idx+i].sim_cnt;
      for(j=0; j<BYTE_SIZE; j++)
        detect[lthread_idx].ent_cnt[j] += detect[lthread_idx+i].ent_cnt[j];
    }
    __syncthreads();
  }

  res[blockIdx.x]->sim_cnt = detect[0].sim_cnt;
  res[blockIdx.x]->ent_v=0;
  for(j=0; j<BYTE_SIZE; j++)
  {
      Pb = ((double)detect[lthread_idx].ent_cnt[j]) / PG_SIZE;
      //Pb = 16.0/PG_SIZE;//1.0/256;
      if(Pb!=0)   res[blockIdx.x]->ent_v += Pb * (-log2(Pb));
     // printf("%lf, %lf\n", ent_v,Pb);
  } 

}

int main(void)
{
  int data_size = 1<<30;  //1GB
  char *data_old, *data_new;
  char *d_do, *d_dn;
  int num_threads;
  int num_blocks;
  SIM_ENT *res, *d_res;
  double start, finish;

  //데이터 초기화
  data_old = (char*)malloc(data_size*sizeof(char)); //1GB
  data_new = (char*)malloc(data_size*sizeof(char)); //1GB
  res = (SIM_ENT*)malloc((data_size/PG_SIZE)*sizeof(SIM_ENT)); //256K*12B = 3KB

  //디바이스 데이터 할당
  cudaMalloc(&d_do, data_size*sizeof(char));  //1GB
  cudaMalloc(&d_dn, data_size*sizeof(char));  //1GB
  cudaMalloc(&d_res, (data_size/PG_SIZE) * sizeof(SIM_ENT)); //3KB

  //데이터 초기화
  for (int i = 0; i < data_size; i++) {
    data_old[i] = 1;
    data_new[i] = 2;
  }

  printf("Start Evaluation\n");
  printf("1. memcpy\n");

  GET_TIME(start);
  cudaMemcpy(d_do, data_old, data_size*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dn, data_new, data_size*sizeof(char), cudaMemcpyHostToDevice);

  //block size 설정 (조정가능)
  num_threads = 32;
  num_blocks = data_size/num_threads; //1GB/32?? 상관무
  
  // Perform sim_ent function on 1GB elements
  //3번째인자는 shared memory의 크기 = 516Bytes * 32 = 16KB (L1cache size 48KB)
  sim_ent<<<num_blocks, num_threads, num_threads*sizeof(BLK_DET)>>>(data_size, d_do, d_dn, d_res);

  cudaMemcpy(res, d_res, (data_size/PG_SIZE) * sizeof(SIM_ENT), cudaMemcpyDeviceToHost);

  GET_TIME(finish);
  printf("Elapsed time = %e seconds\n", finish - start);

  cudaFree(d_do);
  cudaFree(d_dn);
  cudaFree(d_res);
  
  free(data_old);
  free(data_new);
  free(res);
}