120180389 안진우 분산 프로그래밍 최종 프로젝트

프로그램은 다음과 같은 순서로 실행한다.

1. 프로그램을 컴파일 한다.
$make

컴파일시 생성되는 프로그램은 다음과 같다.

CPU_sim: 단일 쓰레드 기반의 데이터 similarity 탐지
CPU_ent: 단일 쓰레드 기반의 데이터 entropy 탐지
CPU_sim_ent: 단일 쓰레드 기반의 데이터 similarity, entropy 동시 탐지

GPU_sim: GPU 기반의 데이터 similarity 탐지
GPU_ent: GPU 기반의 데이터 entropy 탐지
GPU_sim_ent: GPU 기반의 데이터 similarity, entropy 동시 탐지

2. 프로그램을 실행한다.
$./CPU_sim
$./CPU_ent
$./CPU_sim_ent
$./GPU_sim 32 //인자로 block size를 직접 적는다. 
$./GPU_ent 32 
$./GPU_sim_ent 32 




