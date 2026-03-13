from pathlib import Path
import yaml
from dataclasses import asdict


def init_paths(params, rank):

    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()  # simulations/ -> project root

    base = PROJECT_ROOT / "data/shear" / params.case_name
    

    paths = {
        "base": base,
        "logs": base / "logs",
        "dump": base / "dump",
        "restart": base / "restart",
        "metadata": base / "metadata.yaml",
    }

    if rank == 0:
        for p in paths.values():
            if not p.suffix:
                p.mkdir(parents=True, exist_ok=True)

    return paths


def save_metadata(params, paths):
    """Save simulation parameters to metadata.yaml."""
    metadata = asdict(params)

    # Convert all Path objects to strings
    for key, value in metadata.items():
        if isinstance(value, Path):
            metadata[key] = str(value)

    # Add provenance info
    metadata["provenance"] = {
        "work_dir": str(paths["base"])
    }

    with open(paths["metadata"], "w") as f:
        yaml.safe_dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"--- Metadata saved to {paths['metadata']} ---", flush=True)