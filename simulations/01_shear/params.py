# params.py
# python simulations/01_shear/run.py --temperature 800 --strain_rate 1e7 --input /home/Ethan/Projects/prec_interactions/input/Fe_E111_110_R30.lmp --potential /home/Ethan/Projects/prec_interactions/potentials/mendelev03.fs

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class SimParams:
    """Simulation parameters for LAMMPS runs."""

    # --- Required ---
    temperature: int
    strain_rate: int
    input: Path
    run_time: int
    potential_path: Path

    thermo_time: int
    ramp_time: int

    # --- Optional / user ---
    name: str | None = None
    bench: int | None = None  # 0 = test, 1 = bench

    # --- Simulation defaults ---
    dt: float = 0.001
    
    species: str = "Fe"

    random_seed: int = field(default_factory=lambda: np.random.randint(1000, 10_000))
    num_cores: int = 1

    def __post_init__(self):
        """Adjust parameters based on bench mode."""
        if self.bench == 0:  # Test mode
            self.run_time = 1
            self.thermo_time = 1
            self.ramp_time = 1
        elif self.bench == 1:  # Benchmark mode
            self.run_time = 500
            self.thermo_time = 500
            self.ramp_time = 500

    @property
    def case_name(self) -> str:
        """Generate a folder name based on bench mode, custom name, or default."""
        if self.name is not None:
            return self.name

        if self.bench == 0:
            prefix = "test"
        elif self.bench == 1:
            prefix = "bench"
        else:
            prefix = "shear"

        return f"{prefix}_T{self.temperature}_S{self.strain_rate:.0e}_N{self.random_seed}"


# =========================
# Argument parser
# =========================
def parse_arguments() -> SimParams:
    """Parse command-line arguments and return a SimParams object."""
    import argparse

    parser = argparse.ArgumentParser(description="MD simulation runner")

    parser.add_argument("--temperature", type=int, required=True, help="Simulation temperature (K)")
    parser.add_argument("--strain_rate", type=float, required=True, help="Strain rate")
    parser.add_argument("--input", type=str, required=True, help="Path to LAMMPS input data file")
    parser.add_argument("--name", type=str, default=None, help="Custom name for the simulation")
    parser.add_argument("--bench", type=int, choices=[0, 1], default=None, help="Benchmark mode: 0=test, 1=bench")
    parser.add_argument("--run_time", type=int, default=100_000, help="Simulation length")
    parser.add_argument("--potential", type=str, required=True, default=None, help="Absolute path to potential path")
    parser.add_argument("--thermo_time", type=int, required=False, default=20_000, help="Time to thermalise the system")
    parser.add_argument("--ramp_time", type=int, required=False, default=20_000, help="Time to ramp the system")

    parser.add_argument(
        "--random_seed",
        type=int,
        default=np.random.randint(1000, 10_000),
        help="Optional random seed for velocities (default=random)",
    )

    args = parser.parse_args()

    return SimParams(
        temperature=args.temperature,
        strain_rate=args.strain_rate,
        input=Path(args.input).resolve(),
        name=args.name,
        bench=args.bench,
        random_seed=args.random_seed,
        run_time=args.run_time,
        potential_path=Path(args.potential).resolve(),
        thermo_time=args.thermo_time,
        ramp_time=args.ramp_time
    )