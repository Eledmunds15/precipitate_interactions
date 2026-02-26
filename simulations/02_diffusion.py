"""sim_framework.py

General framework for running MD simulations.
Extend by subclassing SimParams and implementing run_simulation().

Usage:
    mpirun -np 4 python simulations/01_shear.py --temperature 100 --strain_rate 1e7 --input /home/Ethan/Projects/prec_interactions/input/Fe_E111_110_R20.lmp
"""

import argparse, re, yaml
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from mpi4py import MPI


# ==============================================================================
# 1. Parameters
#    - Add new parameters here as new fields
#    - Override defaults per simulation type in a subclass
# ==============================================================================

@dataclass
class SimParams:
    # --- CLI arguments ---
    temperature: int
    strain_rate: float
    input: Path
    bench: int = None
    run_time: int = 1_500_000

    # --- Defaults ---
    dt: float = 0.001
    species: str = "Fe"
    potential_path: Path = Path("potentials/mendelev03.fs")
    initial_dislo_dist: int = 40
    thermo_freq: int = 100
    dump_freq: int = 1000
    restart_freq: int = 10_000
    
    # Initialize these so they can be overwritten
    thermo_time: int = 10_000
    ramp_time: int = 10_000

    # --- Derived ---
    radius: int = field(init=False)
    num_cores: int = field(default=1) # We will update this in main()
    random_seed: int = field(default_factory=lambda: np.random.randint(1000, 10_000))

    def __post_init__(self):
        # 1. Extract Radius
        match = re.search(r"_R(\d+)", self.input.stem)
        if not match:
            raise ValueError(f"Could not extract radius from filename: {self.input.name}")
        self.radius = int(match.group(1))

        # 2. Benchmarking Logic: Override run times
        if self.bench == 0:
            self.thermo_time = 1
            self.ramp_time = 1
            self.run_time = 1
        elif self.bench == 1:
            self.thermo_time = 500
            self.ramp_time = 500
            self.run_time = 500

    @property
    def case_name(self) -> str:
        """Dynamic directory name based on bench status."""
        if self.bench == 0:
            return "TEST"
        
        sr = f"{self.strain_rate:.0E}".replace("E+0", "E").replace("E+", "E")
        standard_name = (
            f"shear_T{self.temperature}"
            f"_SR{sr}"
            f"_R{self.radius}"
            f"_N{self.random_seed}"
        )

        if self.bench == 1:
            return f"bench_{standard_name}_NUM{self.num_cores}"
        
        return standard_name


# ==============================================================================
# 2. Paths
#    - Add new output paths here as needed
# ==============================================================================

def init_paths(params: SimParams, rank: int) -> dict[str, Path]:
    """Build output paths and create the case directory (rank 0 only)."""
    base = Path("data") / params.case_name

    paths = {
        "base":     base,
        "metadata": base / "metadata.yaml",
        "logs":     base / "logs",
        "dump":    base / "dump",
        "restart": base / "restart",
        "output": base / "output"
    }

    for p in paths.values():
        if not p.suffix:
            p.mkdir(parents=True, exist_ok=True)

    return paths


# ==============================================================================
# 3. Metadata
#    - Extend the metadata dict if you add new parameter groups
# ==============================================================================

def save_metadata(params: SimParams, paths: dict[str, Path]) -> None:
    """Save simulation parameters to metadata.yaml."""
    
    metadata = asdict(params)

    # Convert all Path objects to strings
    for key, value in metadata.items():
        if isinstance(value, Path):
            metadata[key] = str(value)

    metadata["provenance"] = {
        "work_dir": str(paths["base"])
    }

    with open(paths["metadata"], "w") as f:
        yaml.safe_dump(
            metadata,
            f,
            default_flow_style=False,
            sort_keys=False
        )

    print(f"--- Metadata saved to {paths['metadata']} ---", flush=True)


# ==============================================================================
# 4. Argument Parser
#    - Add new arguments here to match new SimParams fields
# ==============================================================================

def parse_arguments() -> SimParams:
    parser = argparse.ArgumentParser(description="MD simulation runner")
    parser.add_argument("--temperature",  type=int,   required=True)
    parser.add_argument("--stress",  type=float, required=True)
    parser.add_argument("--input",        type=str,   required=True)
    parser.add_argument("--pop_file")
    parser.add_argument("--run_time",     type=int,   required=False)
    parser.add_argument("--bench",        type=int,   choices=[0, 1], default=None)
    args = parser.parse_args()

    params = SimParams(
        temperature=args.temperature,
        strain_rate=args.strain_rate,
        input=Path(args.input).resolve(),
        run_time=args.run_time if args.run_time is not None else 1_000_000,
        bench=args.bench,
    )

    return params


# ==============================================================================
# 5. Simulation
#    - Replace the body of this function with your LAMMPS logic
# ==============================================================================

def run_simulation(params: SimParams, paths: dict[str, Path], comm) -> None:
    """Run the LAMMPS simulation. Add your simulation logic here."""
    from lammps import lammps

    lmp = lammps(comm=comm)

    lmp.cmd.log(paths["logs"] / "setup.log") # Set the log path (1 log for the entire simulation)

    lmp.cmd.units("metal")
    lmp.cmd.dimension(3)
    lmp.cmd.boundary("p", "s", "p") # Set shrink-wrapped boundaries along the y-direction
    lmp.cmd.atom_style("atomic")
    lmp.cmd.atom_modify("map", "yes")
    lmp.cmd.timestep(params.dt)

    # ===========================
    # Define the interatomic potential
    # ===========================
    lmp.cmd.read_restart(params.input)
    

# ==============================================================================
# 6. Main
# ==============================================================================

def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    params = parse_arguments()

    # Broadcast random seed
    params.random_seed = comm.bcast(params.random_seed, root=0)
    params.num_cores = size

    paths = init_paths(params, rank)
    comm.Barrier()

    if rank == 0:
        save_metadata(params, paths)

    run_simulation(params, paths, comm)


if __name__ == "__main__":
    main()