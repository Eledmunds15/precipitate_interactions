import time
from pathlib import Path
from mpi4py import MPI
import numpy as np
import shutil

from helpers import print_metadata, section_break, show_processes, unwrap_dislocation_trajectory
from ovito_processing import perform_dxa

def main():
    overall_start = time.time()

    # ==========================
    # MPI setup
    # ==========================
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ==========================
    # Parse arguments
    # ==========================
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    input_dir_path = Path(args.input).resolve()
    output_dir_path = input_dir_path
    case_name = input_dir_path.name

    # ==========================
    # Print metadata (rank 0)
    # ==========================
    if rank == 0:
        print_metadata(input_dir_path, output_dir_path, size)

    comm.barrier()

    # ==========================
    # Create output directories (all ranks)
    # ==========================
    dirs = {
        "dxa_verts": output_dir_path / "dxa_verts",
        "dxa_atoms": output_dir_path / "dxa_atoms",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    # ==========================
    # Distribute input files
    # ==========================
    input_dump_dir_path = input_dir_path / "dump"
    full_list_of_files = sorted(input_dump_dir_path.glob("*.lammpstrj"))  # sorted for consistent ordering
    local_files = full_list_of_files[rank::size]

    # Rank-ordered print
    comm.barrier()
    show_processes(rank, size, comm, local_files)

    if rank == 0:
        section_break()
    comm.barrier()

    # ==========================
    # Process files locally
    # ==========================
    all_verts_local = []
    all_lengths_local = []
    box_dims = None  # will be set from first processed file

    for i, file in enumerate(local_files):
        lengths, verts, file_box_dims = perform_dxa(file, dirs)
        all_lengths_local.append(lengths)
        all_verts_local.append(verts)
        if box_dims is None and file_box_dims is not None:
            box_dims = file_box_dims
        print(f"Rank {rank} processed file {i+1}/{len(local_files)}: {file.name}", flush=True)

    # Stack local arrays
    all_lengths_local = np.vstack(all_lengths_local) if all_lengths_local else np.empty((0, 3))
    all_verts_local   = np.vstack(all_verts_local)   if all_verts_local   else np.empty((0, 6))

    # ==========================
    # Gather arrays to rank 0
    # ==========================
    gathered_lengths  = comm.gather(all_lengths_local, root=0)
    gathered_verts    = comm.gather(all_verts_local,   root=0)
    gathered_box_dims = comm.gather(box_dims,          root=0)

    if rank == 0:
        # Stack arrays from all ranks
        all_lengths = np.vstack(gathered_lengths)  # (total_lines, 3)     -> [timestep, dislo_id, length]
        all_verts   = np.vstack(gathered_verts)    # (total_vertices, 6)  -> [timestep, dislo_id, vertex_id, x, y, z]

        # Use first available box dims
        box_dims = next(b for b in gathered_box_dims if b is not None)

        print(f"Collected {all_lengths.shape[0]} lines and {all_verts.shape[0]} vertices in total", flush=True)

        # ==========================
        # Unwrap dislocation trajectories across frames
        # all_verts cols: [timestep, dislo_id, vertex_id, x, y, z]
        # ==========================
        for dislo_id in np.unique(all_verts[:, 1]):
            id_mask = all_verts[:, 1] == dislo_id
            timesteps = np.unique(all_verts[id_mask, 0])

            # Centroid of each dislocation line at each timestep
            centroids = {
                t: all_verts[id_mask & (all_verts[:, 0] == t), 3:6].mean(axis=0)
                for t in timesteps
            }

            # Unwrap the centroid trajectory across frames
            unwrapped = unwrap_dislocation_trajectory(centroids, box_dims)

            # Shift all vertices by the same PBC correction as their centroid
            for t in timesteps:
                t_mask = id_mask & (all_verts[:, 0] == t)
                offset = unwrapped[t] - centroids[t]
                all_verts[t_mask, 3:6] += offset

        # ==========================
        # Save unified arrays
        # ==========================
        np.save(output_dir_path / "dislocation_lengths.npy", all_lengths)
        np.save(output_dir_path / "dislocation_verts.npy",   all_verts)

        print(f"Saved unified arrays to {output_dir_path}", flush=True)

    comm.barrier()
    if rank == 0:
        print(f"All ranks finished in {time.time() - overall_start:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()