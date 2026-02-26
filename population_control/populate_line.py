import argparse
import re
from pathlib import Path
import numpy as np
import random

# ==========================================================
# GLOBAL SETTINGS
# ==========================================================
NUM_ATOMS = 100             
FIXED_SURFACE_DEPTH = 40   # Å (Top/Bottom Y-slabs)
PREC_RADIUS = 30           # Å (Central Sphere)

CLOUD_MIN_DIST = 20.0      # Å (Avoid core overlap)
CLOUD_MAX_DIST = 100.0      # Å (Capture radius)
MIN_SIA_SPACING = 35.0     # Å (Inter-atom spacing)
# ==========================================================

def set_paths(args):
    base_dir = Path("data").resolve()
    for sim_folder in base_dir.glob("sim_T*"):
        match = re.search(r'T(\d+)', sim_folder.name)
        if match and int(match.group(1)) == args.temperature:
            input_file = sim_folder / "output" / "reference.dump"
            dxa_file = sim_folder / "dxa" / f"dxa_{args.restart_id}"
            return sim_folder.resolve(), input_file.resolve(), dxa_file.resolve()
    return None, None, None

def extract_box_bounds(file_path):
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Could not find file: {path}")
    bounds = []
    with path.open('r') as f:
        for line in f:
            if "ITEM: BOX BOUNDS" in line:
                for _ in range(3):
                    line = next(f)
                    low, high = map(float, line.split())
                    bounds.append((low, high))
                break
    return bounds

def get_dxa_vertices(filepath):
    path = Path(filepath).expanduser()
    vertices = []
    origin = np.array([0.0, 0.0, 0.0])
    lengths = np.array([0.0, 0.0, 0.0])
    
    with path.open('r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    for i, line in enumerate(lines):
        if line.startswith("SIMULATION_CELL_ORIGIN"):
            origin = np.array([float(x) for x in line.split()[1:]])
        if line.startswith("SIMULATION_CELL_MATRIX"):
            lengths = np.array([float(lines[i+1].split()[0]), 
                                float(lines[i+2].split()[1]), 
                                float(lines[i+3].split()[2])])
        if line.startswith("DISLOCATIONS"):
            num_dislo = int(line.split()[1])
            current_idx = i + 1
            for d in range(num_dislo):
                num_verts = int(lines[current_idx + 3])
                coord_start = current_idx + 4
                for v_offset in range(num_verts):
                    coords = np.array([float(x) for x in lines[coord_start + v_offset].split()])
                    for dim in [0, 2]:
                        coords[dim] = ((coords[dim] - origin[dim]) % lengths[dim]) + origin[dim]
                    vertices.append(coords)
                current_idx = coord_start + num_verts
    return np.array(vertices)

def is_forbidden(pos, bounds, vertices):
    x, y, z = pos
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])

    if y < (y_min + FIXED_SURFACE_DEPTH) or y > (y_max - FIXED_SURFACE_DEPTH):
        return True

    dist_prec_sq = np.sum((pos - center)**2)
    if dist_prec_sq < PREC_RADIUS**2:
        return True

    dists_sq = np.sum((vertices - pos)**2, axis=1)
    min_dist = np.sqrt(np.min(dists_sq))

    if min_dist < CLOUD_MIN_DIST or min_dist > CLOUD_MAX_DIST:
        return True

    return False

def generate_dislocation_cloud(output_dir, bounds, vertices, num_atoms, args):
    # Dynamic filename using restart_id

    spatial_meta = f"R{PREC_RADIUS}_Min{int(CLOUD_MIN_DIST)}_Max{int(CLOUD_MAX_DIST)}_Sp{MIN_SIA_SPACING}"
    
    # Final filename construction
    filename = f"focused_ID{args.restart_id}_T{args.temperature}_N{num_atoms}_{spatial_meta}.txt"

    out_file = output_dir / filename
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
    lx, lz = x_max - x_min, z_max - z_min
    
    # Calculate segments and lengths for weighted selection
    segments = []
    seg_lengths = []
    for i in range(len(vertices) - 1):
        p1, p2 = vertices[i], vertices[i+1]
        dist = np.linalg.norm(p2 - p1)
        if dist > 0.1: # Skip overlapping vertices
            segments.append((p1, p2))
            seg_lengths.append(dist)
    
    # Add final segment for closed loops/PBC if needed
    dist_loop = np.linalg.norm(vertices[-1] - vertices[0])
    if dist_loop > 0.1:
        segments.append((vertices[-1], vertices[0]))
        seg_lengths.append(dist_loop)

    # Convert lengths to probability weights
    total_l = sum(seg_lengths)
    weights = [l / total_l for l in seg_lengths]

    placed_atoms = []
    max_attempts = num_atoms * 1000
    attempts = 0
    last_reported_progress = -1

    print(f"Generating SIAs for Restart {args.restart_id} (Spacing: {MIN_SIA_SPACING}Å)")

    with out_file.open('w') as f:
        while len(placed_atoms) < num_atoms and attempts < max_attempts:
            attempts += 1
            
            # Weighted random selection of a segment
            seg_start, seg_end = random.choices(segments, weights=weights, k=1)[0]
            anchor = seg_start + random.random() * (seg_end - seg_start)
            
            r = random.uniform(CLOUD_MIN_DIST, CLOUD_MAX_DIST)
            phi = random.uniform(0, 2 * np.pi)
            costheta = random.uniform(-1, 1)
            theta = np.arccos(costheta)
            
            offset = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])
            
            target = anchor + offset
            target[0] = ((target[0] - x_min) % lx) + x_min
            target[2] = ((target[2] - z_min) % lz) + z_min
            
            if not is_forbidden(target, bounds, vertices):
                if all(np.linalg.norm(target - p) >= MIN_SIA_SPACING for p in placed_atoms):
                    f.write(f"{target[0]:.4f} {target[1]:.4f} {target[2]:.4f}\n")
                    placed_atoms.append(target)
            
            progress = int((len(placed_atoms) / num_atoms) * 100)
            if progress % 20 == 0 and progress != last_reported_progress:
                print(f"  Progress: {progress}% ({len(placed_atoms)}/{num_atoms})")
                last_reported_progress = progress

    if len(placed_atoms) < num_atoms:
        print(f"!! Warning: Only placed {len(placed_atoms)}/{num_atoms}. Spacing might be too high.")
    else:
        print(f"Done! Created: {out_file.name}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dislocation Restart Minimization")
    parser.add_argument("--temperature", type=int, required=True)
    parser.add_argument("--restart_id", type=int, required=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    sim_folder, input_file, dxa_file = set_paths(args)

    if not sim_folder or not dxa_file.exists():
        print(f"Error: Missing DXA file at {dxa_file}")
        return

    box_bounds = extract_box_bounds(input_file)
    dxa_verts = get_dxa_vertices(dxa_file)
    
    output_dir = sim_folder / "rerun" / "inputs"
    generate_dislocation_cloud(output_dir, box_bounds, dxa_verts, NUM_ATOMS, args)

if __name__ == "__main__":
    main()