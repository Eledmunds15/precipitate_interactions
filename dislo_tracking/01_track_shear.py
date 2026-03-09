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
    exp_mod = ExpressionSelectionModifier(expression="(Cluster == 1 || ParticleType == 2 || ParticleType == 3) && ParticleType != 4") 
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
    export_file(pipeline, str(dirs["raw"] / f"dxa_{timestep}.ca"), "ca", export_mesh=False)

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

def process_log(sim_path, output_dir, log_filename="run.log"):
    """
    Robustly extracts multi-line wrapped LAMMPS thermo data from run.log.
    """
    log_file = Path(sim_path) / "logs" / log_filename
    if not log_file.exists():
        print(f"  [LogProcessor] Error: {log_file} not found.", flush=True)
        return

    with open(log_file, 'r') as f:
        content = f.read()

    # 1. Identify the header and the number of columns
    # We look for the line starting with 'Step' and ending with the last specific column
    header_pattern = r"(Step\s+Temp\s+PotEng.*?c_mobstressXZ\s*)\n"
    header_match = re.search(header_pattern, content)
    
    if not header_match:
        print(f"  [LogProcessor] Error: Could not find valid thermo header in {log_filename}")
        return

    header_str = header_match.group(1)
    columns = header_str.split()
    num_cols = len(columns)

    # 2. Extract the raw data block between header and the 'Loop' or 'ERROR'
    # This captures everything until LAMMPS finishes the run block
    data_block_pattern = re.escape(header_str) + r"(.*?)(?=Loop|ERROR|$)"
    data_match = re.search(data_block_pattern, content, re.DOTALL)
    
    if not data_match:
        print(f"  [LogProcessor] Error: Could not find data block after header.")
        return

    raw_data = data_match.group(1)

    # 3. Clean and parse multi-line data
    # We split by all whitespace to get a flat list of all numeric strings.
    # This ignores physical line breaks and treats the data as one long sequence.
    all_values = raw_data.split()
    
    # Filter out common non-numeric artifacts like WARNINGs that might be in the block
    numeric_values = []
    for val in all_values:
        try:
            float(val)
            numeric_values.append(val)
        except ValueError:
            continue # Skip "WARNING:", "Temperature", etc.

    # 4. Reshape the flat list into rows based on the column count
    data_rows = [numeric_values[i:i + num_cols] for i in range(0, len(numeric_values), num_cols)]

    # 5. Build DataFrame and Save
    if data_rows:
        final_df = pd.DataFrame(data_rows, columns=columns)
        # Convert all columns to numeric (floats/ints)
        final_df = final_df.apply(pd.to_numeric, errors='coerce')
        
        # Drop any rows that didn't have enough columns (incomplete final step)
        final_df = final_df.dropna(subset=['Step'])
        
        # Deduplicate based on Step (keeps the last occurrence if a run was restarted)
        final_df = final_df.drop_duplicates(subset=['Step'], keep='last')
        
        output_path = Path(output_dir) / "log_scalars.csv"
        final_df.to_csv(output_path, index=False)
        print(f"  [LogProcessor] Processed {len(final_df)} timesteps into log_scalars.csv", flush=True)
    else:
        print(f"  [LogProcessor] Warning: No valid data rows found.")

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1. Setup Paths
    raw_sim_path = Path(args.input).resolve()
    
    dirs = {
        "raw": raw_sim_path / "dxa_raw",
        "atoms": raw_sim_path / "dxa_atoms",
        "verts": raw_sim_path / "dxa_verts",
        "analysis": raw_sim_path / "analysis"
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
        print(f"Results stored in: {dirs}", flush=True)
        print(f"{'='*70}\n", flush=True)

    if rank != 0:
        print(f"Rank {rank} finished.", flush=True)

if __name__ == "__main__":
    main()