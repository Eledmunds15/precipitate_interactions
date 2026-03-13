from mpi4py import MPI

from params import parse_arguments
from paths import init_paths, save_metadata
from lammps_sim import run_simulation


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    params = parse_arguments()

    params.random_seed = comm.bcast(params.random_seed, root=0)
    params.num_cores = size

    paths = init_paths(params, rank)

    comm.Barrier()

    if rank == 0:
        save_metadata(params, paths)

    run_simulation(params, paths, comm)


if __name__ == "__main__":
    main()