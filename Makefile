all :
	nvcc -o GPU_sim GPU_sim.cu
	nvcc -o GPU_ent GPU_ent.cu
	nvcc -o GPU_sim_ent GPU_sim_ent.cu

	gcc -o CPU_sim CPU_sim.c -lm
	gcc -o CPU_ent CPU_ent.c -lm
	gcc -o CPU_sim_ent CPU_sim_ent.c -lm

clean :
	rm GPU_sim
	rm GPU_ent
	rm GPU_sim_ent
	rm CPU_sim
	rm CPU_ent
	rm CPU_sim_ent

