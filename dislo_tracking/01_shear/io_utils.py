from pathlib import Path
import os
import io
import pandas as pd

def get_dump_files(sim_path):
    """Return sorted list of dump files in a simulation folder."""
    dump_dir = sim_path / "dump"
    return sorted(dump_dir.glob("*.lammpstrj"))

def prepare_dirs(processed_base):
    """Create directory structure for processed outputs."""
    dirs = {
        "raw": processed_base / "dxa_raw",
        "atoms": processed_base / "dxa_atoms",
        "analysis": processed_base / "analysis"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def process_log(sim_path):
    """Extract thermo/log data into a DataFrame."""
    log_file = sim_path / "logs" / "log.lammps"
    if not log_file.exists():
        return None
    all_blocks = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
    is_capturing, current_block = False, []
    for line in lines:
        clean = line.strip()
        if clean.startswith("Step"):
            is_capturing, current_block = True, [clean]
            continue
        if is_capturing:
            if clean.startswith("Loop") or not clean or "ERROR" in clean:
                if len(current_block) > 1:
                    df = pd.read_csv(io.StringIO("\n".join(current_block)), sep=r'\s+')
                    all_blocks.append(df)
                is_capturing = False
            else:
                current_block.append(clean)
    return pd.concat(all_blocks, ignore_index=True) if all_blocks else None