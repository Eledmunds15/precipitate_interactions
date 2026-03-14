import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (
    DislocationAnalysisModifier,
    ExpressionSelectionModifier,
    DeleteSelectedModifier
)

from helpers import wrap_dislocation_line

def perform_dxa(input_file, dirs):

    # ==========================
    # Prepare lists for output
    # ==========================
    verts = []   # Cols: dislo_id, vertex_id, x, y, z
    lengths = [] # Cols: dislo_id, length

    # ==========================
    # Prepare Pipeline
    # ==========================    
    pipeline = import_file(str(input_file))

    dxa_mod = DislocationAnalysisModifier(
        input_crystal_structure=DislocationAnalysisModifier.Lattice.BCC
    )
    exp_mod = ExpressionSelectionModifier(
        expression="(Cluster == 1 || ParticleType == 2 || ParticleType == 3) && ParticleType != 4"
    )
    del_mod = DeleteSelectedModifier()
    
    for mod in [dxa_mod, exp_mod, del_mod]:
        pipeline.modifiers.append(mod)

    data = pipeline.compute()

    # ==========================
    # Collect data
    # ==========================
    timestep = int(data.attributes["Timestep"])
    T = 50  # Number of points along the dislocation line
    cell = data.cell

    # Extract box dims to return alongside arrays
    box_dims = np.array([cell[0,0], cell[1,1], cell[2,2]]) if cell is not None else None

    for line in data.dislocations.lines:
        t_values = np.linspace(0, 1, T, endpoint=False)
        sampled_points = np.array([line.point_along_line(t) for t in t_values])

        # Stitch the line across PBC boundaries (segment-by-segment)
        if cell is not None:
            sampled_points = wrap_dislocation_line(sampled_points, cell)

        for vertex_id, xyz in enumerate(sampled_points):
            verts.append([line.id, vertex_id, *xyz])

        lengths.append([line.id, line.length])

    # ==========================
    # Convert lists to NumPy arrays and prepend timestep
    # ==========================
    if verts:
        verts = np.array(verts)                                                       # (n_verts, 5)
        verts = np.hstack([np.full((verts.shape[0], 1), timestep), verts])            # (n_verts, 6)
    else:
        verts = np.empty((0, 6))

    if lengths:
        lengths = np.array(lengths)                                                   # (n_lines, 2)
        lengths = np.hstack([np.full((lengths.shape[0], 1), timestep), lengths])      # (n_lines, 3)
    else:
        lengths = np.empty((0, 3))

    # ==========================
    # Export Data
    # ==========================
    dxa_verts_path = dirs["dxa_verts"] / f"dxa_atoms_{timestep}.ca"
    dxa_atoms_path = dirs["dxa_atoms"] / f"dxa_atoms_{timestep}.dump"

    export_file(
        data=data,
        file=dxa_verts_path,
        format="ca",
        export_mesh=False
    )

    export_file(
        data=data,
        file=dxa_atoms_path,
        format="lammps/dump",
        columns=[
            "Particle Identifier",
            "Particle Type",
            "Position"
        ]
    )

    # ==========================
    # Return timestep-aware arrays and box dims
    # ==========================
    return lengths, verts, box_dims