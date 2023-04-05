export CCL_SYCL_OUTPUT_EVENT=1
export CCL_USE_EXTERNAL_QUEUE=1

__model_name=bloom-560m
debugger=ipdb3
# debugger="gdbserver :44333 python"

mpirun -np 1 python -m mpi_to_deepspeed --module inference_server.benchmark --model_name $__model_name --model_class AutoModelForCausalLM --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 5 : -np 1 ipdb3 -m mpi_to_deepspeed --module inference_server.benchmark --model_name $__model_name --model_class AutoModelForCausalLM --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 5
