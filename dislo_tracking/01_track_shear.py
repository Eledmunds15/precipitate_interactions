import argparse, io, re, os, time
from pathlib import Path
from mpi4py import MPI
import numpy as np
import pandas as pd

from ovito.io import import_file, export_file
from ovito.modifiers import DislocationAnalysisModifier, ExpressionSelectionModifier, DeleteSelectedModifier 

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel DXA and Log Processor")
    parser.add_argument("--input", type=str, required=True, help="Path to raw simulation directory")
    return parser.parse_args()

def natural_key(string_):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', str(string_))]

def process_dump(input_file, dirs):
    """
    Handles DXA, atom filtering, and extracting a standardized Z-grid trajectory.
    """
    pipeline = import_file(str(input_file))
    
    # 1. Setup Modifiers
    dxa_mod = DislocationAnalysisModifier(input_crystal_structure=DislocationAnalysisModifier.Lattice.BCC)
    exp_mod = ExpressionSelectionModifier(expression="Cluster == 1 || ParticleType == 2 || ParticleType == 3") 
    del_mod = DeleteSelectedModifier()

    pipeline.modifiers.append(dxa_mod)
    pipeline.modifiers.append(exp_mod)
    pipeline.modifiers.append(del_mod)

    data = pipeline.compute()
    timestep = int(data.attributes["Timestep"])
    lz = data.cell[2,2]

    # 2. Exports
    export_file(pipeline, str(dirs["atoms"] / f"dxa_atoms_{timestep}.dump"), 
                "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position", "Velocity", "Force", "Stress Tensor", "c_pe", "c_ke"])
    export_file(pipeline, str(dirs["raw"] / f"dxa_{timestep}.ca"), "ca")

    # 3. Standardized Grid (Z-Resampling)
    stats = {"timestep": timestep, "length": 0.0}
    
    if data.dislocations.segments:
        segment = max(data.dislocations.segments, key=lambda s: s.length)
        N = 50
        t_values = np.linspace(0, 1, N * 2) 
        raw_points = np.array([segment.point_along_line(t) for t in t_values])
        
        z_raw = raw_points[:, 2] % lz
        x_raw = raw_points[:, 0]
        y_raw = raw_points[:, 1]
        
        idx = np.argsort(z_raw)
        z_sort, x_sort, y_sort = z_raw[idx], x_raw[idx], y_raw[idx]
        
        z_uniform = np.linspace(0, lz, N)
        x_resampled = np.interp(z_uniform, z_sort, x_sort, period=lz) 
        y_resampled = np.interp(z_uniform, z_sort, y_sort, period=lz)
        
        df = pd.DataFrame({'z': z_uniform, 'x': x_resampled, 'y': y_resampled})
        df.to_csv(dirs["verts"] / f"dxa_{timestep}.csv", index=False)
        stats["length"] = segment.length

    return stats

def process_log(sim_path, output_dir):
    log_file = os.path.join(sim_path, "log.lammps")
    if not os.path.exists(log_file):
        print(f"  [LogProcessor] Error: {log_file} not found.", flush=True)
        return

    all_data_frames = []
    with open(log_file, 'r') as f:
        lines = f.readlines()

    is_capturing = False
    current_block = []

    for line in lines:
        clean_line = line.strip()
        if clean_line.startswith("Step"):
            is_capturing = True
            current_block = [clean_line]
            continue
        
        if is_capturing:
            if not clean_line or clean_line.startswith("Loop") or clean_line.startswith("ERROR"):
                if current_block:
                    df_block = pd.read_csv(io.StringIO("\n".join(current_block)), sep=r'\s+')
                    all_data_frames.append(df_block)
                is_capturing = False
                current_block = []
            else:
                current_block.append(clean_line)

    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "log_scalars.csv")
        final_df.to_csv(output_file, index=False)
        print(f"  [LogProcessor] Successfully processed {len(all_data_frames)} blocks into {output_file}", flush=True)
    else:
        print("  [LogProcessor] No thermodynamic data blocks found.", flush=True)

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Setup Paths
    raw_sim_path = Path(args.input).resolve()
    processed_base = raw_sim_path.parents[1] / "processed" / raw_sim_path.name
    
    dirs = {
        "raw": processed_base / "dxa_raw",
        "atoms": processed_base / "dxa_atoms",
        "verts": processed_base / "dxa_verts",
        "analysis": processed_base / "analysis"
    }

    # 2. Rank 0: Initialization
    start_time = time.time()
    if rank == 0:
        print(f"\n{'='*70}", flush=True)
        print(f"MPI DXA PROCESSOR STARTING", flush=True)
        print(f"Target Directory: {raw_sim_path}", flush=True)
        print(f"Number of Ranks: {size}", flush=True)
        print(f"{'='*70}\n", flush=True)
        
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        dump_dir = raw_sim_path / "dump"
        all_files = sorted(list(dump_dir.glob("*.lammpstrj")), key=natural_key)
        chunks = np.array_split(all_files, size)
        print(f"Master: Distributing {len(all_files)} files.", flush=True)
    else:
        chunks = None

    my_files = comm.scatter(chunks, root=0)
    num_my_files = len(my_files)

    # 3. Parallel Processing
    local_stats = []
    for i, f in enumerate(my_files):
        # Time the individual frame
        f_start = time.time()
        result = process_dump(f, dirs)
        f_end = time.time()
        
        if result:
            local_stats.append(result)
            length_val = f"{result['length']:.2f}"
        else:
            length_val = "N/A"
            
        # Print progress for every file
        print(f"[Rank {rank:02d}] {i+1:03d}/{num_my_files:03d} | File: {f.name} | Len: {length_val} | Time: {f_end-f_start:.2f}s", flush=True)

    # Gather results from all ranks to Rank 0
    all_gathered_stats = comm.gather(local_stats, root=0)

    # Ensure all ranks finish before starting post-processing
    comm.Barrier()

    # 4. Final Analysis & Visualisation
    if rank == 0:
        # Flatten the list of lists gathered via MPI
        flat_stats = [item for sublist in all_gathered_stats if sublist for item in sublist]
        
        if flat_stats:
            df_stats = pd.DataFrame(flat_stats).sort_values("timestep")
            df_stats.to_csv(dirs["analysis"] / "simulation_summary.csv", index=False)
            print(f"\nMaster: Saved simulation_summary.csv ({len(df_stats)} frames).", flush=True)

        print("Master: Starting log processing...", flush=True)
        process_log(raw_sim_path, dirs["analysis"])
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}", flush=True)
        print(f"PROCESSING COMPLETE", flush=True)
        print(f"Total Wall Time: {total_time:.2f} seconds", flush=True)
        print(f"Results stored in: {processed_base}", flush=True)
        print(f"{'='*70}\n", flush=True)

    if rank != 0:
        print(f"Rank {rank} finished.", flush=True)

if __name__ == "__main__":
    main()