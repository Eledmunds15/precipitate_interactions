"""
sim_framework.py
Molecular Dynamics (MD) Simulation Framework for Stress-Controlled Diffusion.

ENVIRONMENT LOGIC:
    - Root is dynamically determined via 'prec_interactions' folder name.
    - All input paths in metadata.json are relative to the project 'data' directory.
    - Results are written to: .../data/diffusion_T{temp}_ST{stress}_R{radius}_N{seed}/
"""

import numpy as np
import json
import yaml
import shutil
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from mpi4py import MPI

@dataclass
class SimParams:
    radius: int
    target_stress: float
    step: int
    temperature: int
    num_sias: int
    box_bounds: dict
    # Re-anchored absolute paths
    restart_file: Path
    atoms_file: Path
    original_sim_dir: Path
    potential_path: Path
    # Simulation settings
    bench: int = None
    thermo_time: int = 10_000
    run_time: int = 1_500_000
    dt: float = 0.001
    species: str = "Fe"
    thermo_freq: int = 100
    dump_freq: int = 1000
    restart_freq: int = 10_000
    num_cores: int = field(default=1)
    random_seed: int = field(default_factory=lambda: 1234)

    def __post_init__(self):
        if self.bench == 0:
            self.thermo_time = 1
            self.run_time = 1
        elif self.bench == 1:
            self.thermo_time = 500
            self.run_time = 500

    @property
    def case_name(self) -> str:
        return f"diffusion_T{self.temperature}_ST{int(self.target_stress)}_R{self.radius}_N{self.random_seed}"

def get_project_root() -> Path:
    """Finds the 'prec_interactions' root directory."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == "prec_interactions":
            return parent
    return current_path.parent.parent

def load_metadata(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {file_path}")
    with open(file_path) as f:
        return json.load(f)

def parse_arguments() -> SimParams:
    parser = argparse.ArgumentParser(description="MD simulation runner")
    parser.add_argument("--meta", type=str, required=True, help="Path to metadata JSON")
    parser.add_argument("--bench", type=int, choices=[0, 1], default=None)
    parser.add_argument("--run_time", type=int, default=None)
    args = parser.parse_args()

    meta_path = Path(args.meta).resolve()
    meta = load_metadata(meta_path)
    root = get_project_root()
    data_dir = root / "data"

    return SimParams(
        radius=meta["Radius"],
        target_stress=meta["TargetStress"],
        step=meta["Step"],
        temperature=meta["Temperature"],
        num_sias=meta["NumSIAs"],
        box_bounds=meta["BoxBounds"],
        # Build absolute paths from relative keys in JSON
        restart_file=data_dir / meta["RestartFile_rel"],
        atoms_file=data_dir / meta["AtomsFile_rel"],
        original_sim_dir=data_dir / meta["OriginalSimDir_rel"],
        potential_path=root / "potentials" / "mendelev03.fs",
        bench=args.bench,
        run_time=args.run_time if args.run_time is not None else 1_500_000,
    )

def init_paths(params: SimParams) -> dict:
    # Results are stored in 'rerun_results' relative to the current project root
    root = get_project_root()
    base = root / "data" / params.case_name
    
    paths = {
        "base": base,
        "logs": base / "logs",
        "dump": base / "dump",
        "restart": base / "restart",
        "output": base / "output",
        "inputs": base / "inputs",
        "metadata": base / "run_config.yaml"
    }
    
    # Create directories
    for p in paths.values():
        if not p.suffix: p.mkdir(parents=True, exist_ok=True)

    # Copy files to local run directory for LAMMPS access
    if params.restart_file.exists():
        shutil.copy(params.restart_file, paths["inputs"] / "input.restart")
    else:
        raise FileNotFoundError(f"Restart file missing: {params.restart_file}")

    if params.atoms_file.exists():
        shutil.copy(params.atoms_file, paths["inputs"] / "atoms.txt")
    else:
        raise FileNotFoundError(f"Atoms file missing: {params.atoms_file}")

    return paths

def populate(lmp, filepath):
    """Injects atoms from text file into the LAMMPS instance using string commands."""
    coords = np.loadtxt(filepath)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"--- Populating {len(coords)} SIAs ---")
    
    for row in coords:
        x, y, z = row
        lmp.command(f"create_atoms 1 single {x} {y} {z}")

    lmp.command("group mobgrp type 1")
    lmp.command("group all union fixgrp mobgrp")

def run_simulation(params: SimParams, paths: dict, comm) -> None:
    from lammps import lammps
    lmp = lammps(comm=comm)
    
    # Consistent use of lmp.command() to avoid wrapper errors
    lmp.command(f"log {str(paths['logs'] / 'setup.log')}")
    lmp.command("processors * 2 *")
    lmp.command("units metal")
    lmp.command("dimension 3")
    lmp.command("boundary p s p")
    lmp.command("atom_style atomic")
    lmp.command("atom_modify map yes")
    lmp.command(f"timestep {params.dt}")

    # Read the local copy
    lmp.command(f"read_restart {str(paths['inputs'] / 'input.restart')}")

    # Interatomic Potential
    lmp.command("pair_style eam/fs")
    lmp.command(f"pair_coeff * * {str(params.potential_path)} {params.species} {params.species} {params.species} {params.species}")

    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify delay 10 check yes")

    # Define groups
    lmp.command("group mobgrp type 1")
    lmp.command("group topgrp type 2")
    lmp.command("group botgrp type 3")
    lmp.command("group precgrp type 4")
    lmp.command("group fixgrp union precgrp topgrp botgrp")
    lmp.command("group all union fixgrp mobgrp")

    # Computes
    lmp.command("compute stress all stress/atom NULL")
    lmp.command("compute mobtemp mobgrp temp")
    lmp.command("compute pe all pe/atom")
    lmp.command("compute ke all ke/atom")

    # Energy/Stress Reductions
    for i, component in enumerate(["XX", "YY", "ZZ", "XY", "YZ", "XZ"], 1):
        lmp.command(f"compute mobstress{component} mobgrp reduce sum c_stress[{i}]")
    
    lmp.command("compute mobpetot mobgrp reduce sum c_pe")
    lmp.command("compute mobketot mobgrp reduce sum c_ke")

    # Initial Minimization
    lmp.command("fix 1 fixgrp setforce 0.0 0.0 0.0")
    lmp.command("reset_timestep 0")
    lmp.command("thermo 1")
    lmp.command("thermo_style custom step temp pe pxx pyy pzz")
    lmp.command("thermo_modify temp mobtemp")
    
    lmp.command("min_style cg")
    lmp.command("minimize 1e-4 1e-6 5000 10000")
    lmp.command(f"write_dump all custom {str(paths['output'] / 'minimized_initial.dump')} id type x y z c_pe c_ke c_stress[1] c_stress[2] c_stress[3] c_stress[4] c_stress[5] c_stress[6]")

    # Populate with SIAs from the local inputs directory
    populate(lmp, str(paths["inputs"] / "atoms.txt"))

    # Minimize with SIAs
    lmp.command("reset_timestep 0")
    lmp.command("min_style fire")
    lmp.command("minimize 1e-4 1e-6 2000 10000")
    lmp.command("min_style cg")
    lmp.command("minimize 1e-4 1e-6 5000 10000")
    lmp.command(f"write_dump all custom {str(paths['output'] / 'minimized_final.dump')} id type x y z c_pe c_ke c_stress[1] c_stress[2] c_stress[3] c_stress[4] c_stress[5] c_stress[6]")

    # Setup Simulation Outputs
    lmp.command("reset_timestep 0")
    lmp.command(f"log {str(paths['logs'] / 'diffusion.log')}")
    lmp.command(f"thermo {params.thermo_freq}")
    lmp.command("thermo_style custom step temp pe pxx pyy pzz c_mobtemp c_mobstressXX c_mobstressYY")
    lmp.command("thermo_modify temp mobtemp")

    lmp.command(f"dump dump all custom {params.dump_freq} {str(paths['dump'] / 'dump_*.lammpstrj')} id type x y z c_pe c_stress[1]")
    lmp.command(f"restart {params.restart_freq} {str(paths['restart'] / 'out.*.restart')}")

    # Run Sequence
    lmp.command(f"velocity mobgrp create 10.0 {params.random_seed} mom yes rot yes dist gaussian")
    lmp.command(f"fix 2 mobgrp nvt temp 10.0 {params.temperature} {params.dt*10}")
    lmp.command("fix_modify 2 temp mobtemp")
    lmp.command(f"run {params.thermo_time}")
    
    lmp.command("unfix 2")
    lmp.command(f"fix 2 mobgrp nvt temp {params.temperature} {params.temperature} {params.dt*100}")
    lmp.command("fix_modify 2 temp mobtemp")
    lmp.command(f"run {params.run_time}")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    params = None
    if rank == 0:
        try:
            params = parse_arguments()
        except Exception as e:
            print(f"Rank 0 Error: {e}")
            comm.Abort()

    params = comm.bcast(params, root=0)
    params.num_cores = comm.Get_size()
    
    paths = init_paths(params)
    
    if rank == 0:
        meta_dict = asdict(params)
        for k, v in meta_dict.items():
            if isinstance(v, Path): meta_dict[k] = str(v)
        with open(paths["metadata"], "w") as f:
            yaml.dump(meta_dict, f)

    run_simulation(params, paths, comm)

if __name__ == "__main__":
    main()