#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include "timer.h"

#define PG_SIZE (4096)
#define BYTE_SIZE (256)

void init_entcnt(unsigned short int ent_cnt[BYTE_SIZE]);

int main()
{
    int i,j;
    int data_size =1<<30;//1<<30;  //1GB
    char *data_old, *data_new;
    double start, finish;
    //float *x, *y, *d_x, *d_y;
    data_old = (char*)malloc(data_size*sizeof(char));
    data_new = (char*)malloc(data_size*sizeof(char));

    for (i = 0; i < data_size; i++) {
    data_old[i] = 1;
    data_new[i] = 1;
    }

    unsigned short int sim_v;
    unsigned short int ent_cnt[BYTE_SIZE];
    double ent_v, Pb;

    GET_TIME(start);
    for(i=0; i<data_size/PG_SIZE; i++)
    {
        sim_v=0, ent_v=0;
      //  init_entcnt(ent_cnt);
        for(j=0; j<PG_SIZE; j++)
        {
            if(data_old[j] == data_new[j])  sim_v++;
   //         ent_cnt[data_new[j]]++;
        }
        /*
        for(j=0; j<BYTE_SIZE; j++)
        {
            Pb = ((double)ent_cnt[j]) / PG_SIZE;
            //Pb = 16.0/PG_SIZE;//1.0/256;
            if(Pb!=0)   ent_v += Pb * (-log2(Pb));
           // printf("%lf, %lf\n", ent_v,Pb);

        }
        */
       // printf("%d, %lf\n", sim_v, ent_v);
       //printf("%d\n", sim_v);
    }   
    GET_TIME(finish);
    printf("Elapsed time = %e seconds\n", finish - start);
}

void init_entcnt(unsigned short int ent_cnt[BYTE_SIZE])
{
    for(int i=0; i<BYTE_SIZE; i++)
        ent_cnt[i]=0;
}