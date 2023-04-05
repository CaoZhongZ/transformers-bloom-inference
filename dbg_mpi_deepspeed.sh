export CCL_SYCL_OUTPUT_EVENT=1
export CCL_USE_EXTERNAL_QUEUE=1

__model_name=bloom-560m
# debugger="gdbserver :44333 python"
__debug_node=1

# mpirun -s 1 -np 1 python -m mpi_to_deepspeed --module inference_server.benchmark --model_name $__model_name --model_class AutoModelForCausalLM --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 5 : -np 1 ipdb3 -m mpi_to_deepspeed --module inference_server.benchmark --model_name $__model_name --model_class AutoModelForCausalLM --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 5
mpirun -s $__debug_node -np 2 python -m mpi_to_deepspeed --debug_node $__debug_node --module inference_server.benchmark --model_name $__model_name --model_class AutoModelForCausalLM --dtype fp16 --deployment_framework ds_inference --benchmark_cycles 5
