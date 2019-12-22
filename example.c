#include<stdio.h>

int main()
{
    int i;
    for(i=0; i<10; i++)
    {
        printf("%d\n", (1<<i)-1);
    }
}