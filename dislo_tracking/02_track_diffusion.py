import argparse, re, os, time
from pathlib import Path
from mpi4py import MPI
import numpy as np
import pandas as pd

from ovito.io import import_file, export_file
from ovito.modifiers import DislocationAnalysisModifier, ExpressionSelectionModifier, DeleteSelectedModifier, WignerSeitzAnalysisModifier
from ovito.pipeline import FileSource

def parse_args():
    parser = argparse.ArgumentParser(description="Parallel DXA and Log Processor")
    parser.add_argument("--input", type=str, required=True, help="Path to raw simulation directory")
    return parser.parse_args()

def natural_key(string_):
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', str(string_))]

def process_dump_ws(input_file, reference_file, dirs):
    """
    Handles WS analysis to find vacancies and interstitials.
    Outputs a per-timestep dump file.
    """
    pipeline = import_file(str(input_file))

    ws_mod = WignerSeitzAnalysisModifier()
    ws_mod.reference = FileSource()
    ws_mod.reference.load(str(reference_file))

    exp_mod = ExpressionSelectionModifier(expression="(Occupancy == 1)")
    del_mod = DeleteSelectedModifier()

    for modifier in [ws_mod, exp_mod, del_mod]:
        pipeline.modifiers.append(modifier)

    data = pipeline.compute()
    timestep = int(data.attributes["Timestep"])

    export_file(
        data,
        str(dirs["ws"] / f"ws_{timestep}.dump"),
        "lammps/dump",
        columns=["Particle Identifier", "Particle Type", "Position", "Occupancy"],
    )

def process_dump_dxa(input_file, dirs):
    """
    Handles DXA, atom filtering, and extracting a standardized Z-grid trajectory.
    """
    pipeline = import_file(str(input_file))

    dxa_mod = DislocationAnalysisModifier(input_crystal_structure=DislocationAnalysisModifier.Lattice.BCC)
    exp_mod = ExpressionSelectionModifier(expression="(Cluster == 1 || ParticleType == 2 || ParticleType == 3) && ParticleType != 4")
    del_mod = DeleteSelectedModifier()

    pipeline.modifiers.append(dxa_mod)
    pipeline.modifiers.append(exp_mod)
    pipeline.modifiers.append(del_mod)

    data = pipeline.compute()
    timestep = int(data.attributes["Timestep"])
    lz = data.cell[2, 2]

    # Exports
    export_file(pipeline, str(dirs["atoms"] / f"dxa_atoms_{timestep}.dump"),
                "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position", "c_pe"])
    export_file(pipeline, str(dirs["raw"] / f"dxa_{timestep}.ca"), "ca", export_mesh=False)

    # Standardized Grid (Z-Resampling)
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

        # Handle periodic boundary: duplicate points shifted by ±lz for wrap-around interpolation
        z_ext = np.concatenate([z_sort - lz, z_sort, z_sort + lz])
        x_ext = np.concatenate([x_sort, x_sort, x_sort])
        y_ext = np.concatenate([y_sort, y_sort, y_sort])

        z_uniform = np.linspace(0, lz, N, endpoint=False)
        x_resampled = np.interp(z_uniform, z_ext, x_ext)
        y_resampled = np.interp(z_uniform, z_ext, y_ext)

        df = pd.DataFrame({'z': z_uniform, 'x': x_resampled, 'y': y_resampled})
        df.to_csv(dirs["verts"] / f"dxa_{timestep}.csv", index=False)
        stats["length"] = segment.length

    return stats

def process_dump(input_file, reference_file, dirs):
    """
    Runs DXA then WS sequentially on a single dump file.
    Returns DXA stats dict (WS has no stats to return currently).
    """
    stats = process_dump_dxa(input_file, dirs)
    process_dump_ws(input_file, reference_file, dirs)
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

    header_pattern = r"(Step\s+Temp\s+PotEng.*?c_mobstressXZ\s*)\n"
    header_match = re.search(header_pattern, content)

    if not header_match:
        print(f"  [LogProcessor] Error: Could not find valid thermo header in {log_filename}")
        return

    header_str = header_match.group(1)
    columns = header_str.split()
    num_cols = len(columns)

    data_block_pattern = re.escape(header_str) + r"(.*?)(?=Loop|ERROR|$)"
    data_match = re.search(data_block_pattern, content, re.DOTALL)

    if not data_match:
        print(f"  [LogProcessor] Error: Could not find data block after header.")
        return

    raw_data = data_match.group(1)
    all_values = raw_data.split()

    numeric_values = []
    for val in all_values:
        try:
            float(val)
            numeric_values.append(val)
        except ValueError:
            continue

    data_rows = [numeric_values[i:i + num_cols] for i in range(0, len(numeric_values), num_cols)]

    if data_rows:
        final_df = pd.DataFrame(data_rows, columns=columns)
        final_df = final_df.apply(pd.to_numeric, errors='coerce')
        final_df = final_df.dropna(subset=['Step'])
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

    raw_sim_path = Path(args.input).resolve()

    dirs = {
        "output":   raw_sim_path / "output",
        "raw":      raw_sim_path / "dxa_raw",
        "atoms":    raw_sim_path / "dxa_atoms",
        "verts":    raw_sim_path / "dxa_verts",
        "ws":       raw_sim_path / "wigner_seitz",
        "analysis": raw_sim_path / "analysis",
    }

    reference_file = dirs["output"] / "minimized_initial.dump"

    start_time = time.time()
    if rank == 0:
        print(f"\n{'='*70}", flush=True)
        print(f"MPI DXA PROCESSOR STARTING", flush=True)
        print(f"Target Directory: {raw_sim_path}", flush=True)
        print(f"Number of Ranks: {size}", flush=True)
        print(f"{'='*70}\n", flush=True)

        if not reference_file.exists():
            print(f"ERROR: Reference file not found: {reference_file}", flush=True)
            comm.Abort(1)

        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        dump_dir = raw_sim_path / "dump"
        all_files = sorted(list(dump_dir.glob("*.lammpstrj")), key=natural_key)
        chunks = np.array_split(all_files, size)
        print(f"Master: Distributing {len(all_files)} files across {size} ranks.", flush=True)
    else:
        chunks = None

    my_files = comm.scatter(chunks, root=0)
    num_my_files = len(my_files)

    local_stats = []
    for i, f in enumerate(my_files):
        f_start = time.time()
        result = process_dump(f, reference_file, dirs)
        f_end = time.time()

        if result:
            local_stats.append(result)
            length_val = f"{result['length']:.2f}"
        else:
            length_val = "N/A"

        print(f"[Rank {rank:02d}] {i+1:03d}/{num_my_files:03d} | File: {f.name} | Len: {length_val} | Time: {f_end-f_start:.2f}s", flush=True)

    all_gathered_stats = comm.gather(local_stats, root=0)
    comm.Barrier()

    if rank == 0:
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
        print(f"{'='*70}\n", flush=True)
    else:
        print(f"Rank {rank} finished.", flush=True)

if __name__ == "__main__":
    main()