import argparse
from pathlib import Path
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

    prec_file = str(sim_dir / "output" / "precipitate.dump")
    top_surface_file = str(sim_dir / "output" / "top_surface.dump")
    bottom_surface_file = str(sim_dir / "output" / "bottom_surface.dump")

    # --- Import pipelines ---
    main_pipeline = import_file(data_path)
    prec_pipeline = import_file(prec_file)
    top_surface_file = import_file(top_surface_file)
    bottom_surface_file = import_file(bottom_surface_file)

    # Disable DXA surfaces if using the line-based DXA files
    if use_atoms:
        main_pipeline.modifiers.append(AssignColorModifier(color=(0, 1, 1))) # Cyan color
    else:
        for vis_element in main_pipeline.vis_elements:
            if isinstance(vis_element, SurfaceMeshVis):
                vis_element.enabled = False
    
    # Make the precipitate atoms appear to be Gold color
    prec_pipeline.modifiers.append(AssignColorModifier(color=(1.0, 0.8, 0.2))) # Gold color

    # Get rid of surface atoms...

    # Add pipelines to scene
    main_pipeline.add_to_scene()
    prec_pipeline.add_to_scene()

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
    vp.render_anim(size=(800, 600), background=bg_color, filename=str(render_dir / out_name), fps=20)
    
    # Cleanup scene
    main_pipeline.remove_from_scene()
    prec_pipeline.remove_from_scene()

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

    print("All done!")

if __name__ == "__main__":
    main()