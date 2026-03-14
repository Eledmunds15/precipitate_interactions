from pathlib import Path
import numpy as np

def print_metadata(input_dir, output_dir, size):

    input_dir = Path(input_dir)

    print("=" * 50, flush=True)
    print("RUN CONFIGURATION", flush=True)
    print("=" * 50, flush=True)
    print(f"Input directory : {input_dir}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print("=" * 50, flush=True)

    input_dump_dir = input_dir / "dump"
    files = list(input_dump_dir.glob("*.lammpstrj"))
    no_files_total = len(files)
    no_files_per_rank = no_files_total // size

    print(f"Number of files: {no_files_total}", flush=True)
    print(f"Number of files per rank: {no_files_per_rank}", flush=True)
    print("=" * 50, flush=True)

# helpers.py

def show_processes(rank, size, comm, local_files):
    for r in range(size):
        if rank == r:
            print(f"Rank {rank} processing {len(local_files)} files...", flush=True)
        comm.barrier()

def section_break():

    print("=" * 50, flush=True)

    return None

def wrap_dislocation_line(points, cell):
    """
    Fix a dislocation line that crosses a PBC boundary by walking
    segment by segment and applying minimum image convention.
    """
    Lx, Ly, Lz = cell[0,0], cell[1,1], cell[2,2]
    L = np.array([Lx, Ly, Lz])
    pbc = [cell.pbc[0], cell.pbc[1], cell.pbc[2]]

    wrapped = [points[0].copy()]
    for i in range(1, len(points)):
        delta = points[i] - points[i-1]
        for dim in range(3):
            if pbc[dim]:
                delta[dim] -= np.round(delta[dim] / L[dim]) * L[dim]
        wrapped.append(wrapped[-1] + delta)

    return np.array(wrapped)

def unwrap_dislocation_trajectory(centroids_by_timestep, box_dims):
    
    Lx, Ly, Lz = box_dims[0], box_dims[1], box_dims[2]  # <-- changed from cell[0,0] etc.
    L = np.array([Lx, Ly, Lz])

    timesteps = sorted(centroids_by_timestep.keys())
    unwrapped = {timesteps[0]: centroids_by_timestep[timesteps[0]].copy()}

    for i in range(1, len(timesteps)):
        prev = unwrapped[timesteps[i-1]]
        curr = centroids_by_timestep[timesteps[i]]
        delta = curr - prev
        for dim in range(3):
            delta[dim] -= np.round(delta[dim] / L[dim]) * L[dim]
        unwrapped[timesteps[i]] = prev + delta

    return unwrapped