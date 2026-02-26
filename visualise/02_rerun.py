import argparse, io
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from ovito.io import import_file
from ovito.data import *
from ovito.modifiers import *
from ovito.pipeline import Pipeline
from ovito.vis import Viewport, CoordinateTripodOverlay, SurfaceMeshVis, TextLabelOverlay, PythonViewportOverlay, ViewportOverlayInterface
from ovito.qt_compat import QtCore

# Custom Overlay Class
class TimestepOverlay(ViewportOverlayInterface):
    def __init__(self, step_increment=1000, color=(0,0,0)):
        self.step_increment = step_increment
        self.color = color
        
    def render(self, canvas: ViewportOverlayInterface.Canvas, data: DataCollection, frame: int, **kwargs):
        # 'frame' is provided by the renderer automatically
        timestep = frame * self.step_increment
        canvas.draw_text(f"Timestep: {timestep}", pos=(0.8, 0.2), font_size=0.02, color=self.color)

def set_paths(args):
    base_dir = Path("data").resolve()
    search_pattern = f"**/{args.path}"
    matches = list(base_dir.glob(search_pattern))

    if not matches:
        print(f"[-] Error: No directory found for {search_pattern}")
        return None
    if len(matches) > 1:
        raise FileExistsError(f"Multiple matches found for {search_pattern}!")

    sim_folder = matches[0].resolve()
    print(f"[+] Found simulation directory: {sim_folder.name}")
    return sim_folder

def visualize_simulation(sim_dir, output_subdir, use_atoms=False):
    # Determine file paths based on mode
    if use_atoms:
        data_path = str(sim_dir / "dxa_atoms" / "dxa_atoms_*")
        out_name = "dxa_atoms_anim.mp4"
    else:
        data_path = str(sim_dir / "dxa" / "dxa_*")
        out_name = "dxa_anim.mp4"

    bg_color = (1, 1, 1) # White
    txt_color = (0, 0, 0) # Black

    vacs_file = str(sim_dir / "ws_vacs" / "ws_vac_*")
    sias_file = str(sim_dir / "ws_sias" / "ws_sia_*")

    ## Have to go up to parent to find the precipitate dump files!
    sim_sim_dir = sim_dir.parent.parent
    prec_file = sim_sim_dir / "output" / "precipitate.dump" 

    # --- Import pipelines ---
    main_pipeline = import_file(data_path)
    prec_pipeline = import_file(prec_file)
    vacs_pipeline = import_file(vacs_file)
    sias_pipeline = import_file(sias_file)

    # Disable DXA surfaces if using the line-based DXA files
    if use_atoms:
        main_pipeline.modifiers.append(AssignColorModifier(color=(0, 1, 1))) # Cyan color
    else:
        for vis_element in main_pipeline.vis_elements:
            if isinstance(vis_element, SurfaceMeshVis):
                vis_element.enabled = False
    
    # Make the precipitate atoms appear to be Gold color
    prec_pipeline.modifiers.append(AssignColorModifier(color=(1.0, 0.8, 0.2))) # Gold color
    vacs_pipeline.modifiers.append(AssignColorModifier(color=(0.0, 0.0, 1.0))) # Blue color
    sias_pipeline.modifiers.append(AssignColorModifier(color=(1.0, 0.0, 0.0))) # Red color

    main_pipeline.add_to_scene()
    prec_pipeline.add_to_scene()
    vacs_pipeline.add_to_scene()
    sias_pipeline.add_to_scene()

    # --- Camera & Viewport ---
    vp = Viewport(type=Viewport.Type.Perspective)
    vp.camera_dir = (1, -1, 1)
    vp.camera_up = (0, 1, 0)
    vp.zoom_all()

    # --- Overlays ---
    # Tripod
    tripod = CoordinateTripodOverlay(size=0.07, alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignBottom)
    vp.overlays.append(tripod)

    # Title
    vp.overlays.append(TextLabelOverlay(
        text=f'{sim_dir.name}',
        alignment=QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
        font_size=0.02, text_color=txt_color))

    # Dynamic Timestep
    vp.overlays.append(PythonViewportOverlay(delegate=TimestepOverlay(color=txt_color)))

    # --- Render ---
    render_dir = sim_dir / output_subdir
    render_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {out_name}...")
    vp.render_anim(size=(800, 600), background=bg_color, filename=str(render_dir / out_name), fps=10)
    
    # Cleanup scene
    main_pipeline.remove_from_scene()
    prec_pipeline.remove_from_scene()

def log_info(sim_folder):
    log_file = sim_folder / "logs" / "log.lammps"
    if not log_file.exists():
        print(f"[-] Log file not found at {log_file}")
        return

    print(f"[+] Parsing log file: {log_file}")
    
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find the start of the dynamics run
    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("run"):
            # The actual table starts a few lines after 'run'
            # We look for the line containing 'Step'
            for j in range(i, i + 10):
                if "Step" in lines[j]:
                    start_idx = j
                    break
            if start_idx != -1:
                break

    if start_idx == -1:
        print("[-] Could not find the dynamics data block in the log.")
        return

    # Extract the table lines
    table_data = []
    for line in lines[start_idx:]:
        parts = line.split()
        # Stop if we hit a blank line or a line that doesn't start with a number (except header)
        if not parts: break
        if parts[0] == "Step" or parts[0].isdigit():
            table_data.append(line)
        else:
            if len(table_data) > 1: # We have the header and some data
                break

    # Convert to DataFrame
    df = pd.read_csv(io.StringIO("".join(table_data)), sep=r'\s+')
    
    # Save to CSV
    csv_path = sim_folder / "vis" / "log_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] CSV saved to {csv_path}")

    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1. Temperature
    axes[0].plot(df['Step'], df['Temp'], color='#e74c3c')
    axes[0].set_title('Temperature (K)')
    axes[0].set_ylabel('K')

    # 2. Shear Stress (Pxy)
    axes[1].plot(df['Step'], df['Pxy'], color='#8e44ad')
    axes[1].set_title('Shear Stress (Pxy)')
    axes[1].set_ylabel('Pressure Units')
    axes[1].axhline(0, color='black', lw=1, ls='--') # Zero line for reference

    # 3. Potential Energy
    axes[2].plot(df['Step'], df['PotEng'], color='#2980b9')
    axes[2].set_title('Potential Energy')
    axes[2].set_ylabel('Energy Units')

    # 4. Normal Stresses (Pxx, Pyy, Pzz)
    axes[3].plot(df['Step'], df['Pxx'], label='Pxx', alpha=0.7)
    axes[3].plot(df['Step'], df['Pyy'], label='Pyy', alpha=0.7)
    axes[3].plot(df['Step'], df['Pzz'], label='Pzz', alpha=0.7)
    axes[3].set_title('Normal Stresses')
    axes[3].legend(loc='best', fontsize='small')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Step')

    plt.tight_layout()
    plt.savefig(sim_folder / "vis" / "dynamics_dashboard.png")

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    sim_folder = set_paths(args)
    if sim_folder:
        # Run standard DXA (lines)
        visualize_simulation(sim_folder, "vis", use_atoms=False)
        # Run DXA Atoms (if needed)
        visualize_simulation(sim_folder, "vis", use_atoms=True)

        log_info(sim_folder)

    print("All done!")

if __name__ == "__main__":
    main()