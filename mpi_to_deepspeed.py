import sys
import os
import subprocess
from argparse import ArgumentParser, REMAINDER
TORCH_DISTRIBUTED_DEFAULT_PORT=33443

def parse_args():
    parser = ArgumentParser(description="MPI to DeepSpeed distributed launch"
                            " utility that bridge MPI launching to mimic launching of DeepSpeed")

    parser.add_argument("--master_addr",
                        default="localhost",
                        type=str,
                        help="Master node (rank 0)'s address, should be either"
                        " the IP address or the hostname of node 0, for"
                        " single node multi-proc training, the"
                        " --master_addr can simply be localhost")
    parser.add_argument("--master_port",
                        default=TORCH_DISTRIBUTED_DEFAULT_PORT,
                        type=int,
                        help="Master node (rank 0)'s free port that needs to "
                        "be used for communication during distributed "
                        "training")
    parser.add_argument("--debug_node",
                        default=-1,
                        type=int,
                        help="Would set a ipdb trace to stop the script and open "
                        "interactive debug CLI")
    parser.add_argument("--module",
                        action="store_true",
                        help="Change each process to interpret the launch "
                        "script as a Python module, executing with the same "
                        "behavior as 'python -m'.")
    parser.add_argument("user_script",
                        type=str,
                        help="User script to launch, followed by any required "
                        "arguments.")
    parser.add_argument('user_args', nargs=REMAINDER)
    return parser.parse_args()

def mapping_impi_to_deepspeed(args):
    # local
    os.environ['LOCAL_RANK'] = os.environ['MPI_LOCALRANKID']
    os.environ['LOCAL_WORLD_SIZE'] = os.environ['MPI_LOCALNRANKS']
    # global
    os.environ['RANK'] = os.environ['PMI_RANK']
    os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
    # deepspeed multi-node
    os.environ['CROSS_RANK'] = '0'
    os.environ['CROSS_SIZE'] = '1'
    #
    os.environ['MASTER_ADDR'] = str(args.master_addr)
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['DEBUG_NODE'] = str(args.debug_node)

def main():
    args = parse_args()
    mapping_impi_to_deepspeed(args)
    current_env = os.environ.copy()

    cmd = [sys.executable, '-u']
    if args.module:
        cmd.append('-m')
    cmd += [args.user_script] + args.user_args

    result = subprocess.Popen(cmd, close_fds=False, env=current_env)
    result.wait()

    if result.returncode > 0:
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
