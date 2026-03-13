from lammps import lammps


def run_simulation(params, paths, comm):

    if comm.Get_rank() == 0:
        print("\nStarting shear simulation\n", flush=True)

    comm.Barrier()

    lmp = lammps(comm=comm)

    setup(lmp, params, paths, comm)
    groups(lmp, params, paths)
    computes(lmp, params, paths)
    prepare_outputs(lmp, params, paths)
    thermalise(lmp, params, paths)
    strain_calculations(lmp, params, paths)
    ramp(lmp, params, paths)
    shear(lmp, params, paths)

def setup(lmp, params, paths, comm):
    
    # First create log
    lmp.command(f"log {paths['logs'] / 'log.lammps'}")

    # Settings
    lmp.command(f"units metal")
    lmp.command(f"dimension 3")
    lmp.command(f"boundary p s p")
    lmp.command(f"atom_style atomic")
    lmp.command(f"atom_modify map yes")
    lmp.command(f"timestep {params.dt}")

    if comm.Get_size() > 1: lmp.command(f"processors * 2 *")

    # Input File
    lmp.command(f"read_data {params.input}")
    
    # Interatomic Potential
    lmp.command(f"pair_style eam/fs")
    lmp.command(f"pair_coeff * * {params.potential_path} {params.species} {params.species} {params.species} {params.species}")

    lmp.command(f"neighbor 2.0 bin")
    lmp.command(f"neigh_modify delay 10 check yes")

    return None

def groups(lmp, params, paths):

    lmp.command(f"group mobgrp type 1")
    lmp.command(f"group topgrp type 2")
    lmp.command(f"group botgrp type 3")
    lmp.command(f"group precgrp type 4")
    lmp.command(f"group fixgrp union topgrp botgrp precgrp")
    lmp.command(f"group all union fixgrp mobgrp")

    return None

def computes(lmp, params, paths):

    lmp.command(f"compute mobtemp mobgrp temp")
    lmp.command(f"compute stress mobgrp stress/atom mobtemp virial")

    lmp.command(f"compute pe all pe/atom")
    lmp.command(f"compute ke all ke/atom")

    stress_components = ["XX", "YY", "ZZ", "XY", "YZ", "XZ"]

    for i, comp in enumerate(stress_components):

        lmp.command(f"compute mobstress{comp} mobgrp reduce sum c_stress[{i+1}]")

    lmp.command(f"compute mob_pe mobgrp reduce sum c_pe")
    lmp.command(f"compute mob_ke mobgrp reduce sum c_ke")

    return None

def prepare_outputs(lmp, params, paths):

    boxBounds = lmp.extract_box()

    box_min = boxBounds[0]
    box_max = boxBounds[1]
    xmin, xmax = box_min[0], box_max[0]
    ymin, ymax = box_min[1], box_max[1]
    zmin, zmax = box_min[2], box_max[2]
    
    vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)

    lmp.command(f"reset_timestep 0")

    lmp.command(f"variable n_atoms_mob equal count(mobgrp)")
    lmp.command(f"variable n_atoms equal count(all)")

    lmp.command(f"variable effective_stressXY equal ({0.1}*(c_mobstressXY/{vol})*(v_n_atoms_mob/v_n_atoms))")
    
    # Thermodynamic data
    lmp.command("thermo_style custom " \
        "step temp c_mob_pe c_mob_ke " \
        "pxx pyy pzz pxy pyz pxz " \
        "c_mobstressXX c_mobstressYY c_mobstressZZ " \
        "c_mobstressXY c_mobstressYZ c_mobstressXZ " \
        "v_effective_stressXY"
    )
    lmp.command(f"thermo 100")
    lmp.command(f"thermo_modify temp mobtemp")

    # Dump data
    lmp.command(
        f"dump dump all custom 1000 {paths['dump']}/dump_*.lammpstrj "
        "id type x y z"
    )

    lmp.command(f"restart 1000 {paths['restart'] / '*.restart'}")

    return None

def thermalise(lmp, params, paths):

    lmp.command(f"fix 1 mobgrp nvt temp {params.temperature} {params.temperature} {params.dt*100.0}")
    lmp.command(f"fix_modify 1 temp mobtemp")

    lmp.command(f"fix 2 fixgrp setforce NULL 0.0 NULL")
    lmp.command(f"velocity fixgrp set 0.0 0.0 0.0")

    lmp.command(f"velocity mobgrp create {params.temperature} {params.random_seed} mom yes rot yes dist gaussian")

    lmp.command(f"run {params.thermo_time}")

    return None

def strain_calculations(lmp, params, paths):

    boxBounds = lmp.extract_box()

    box_min = boxBounds[0]
    box_max = boxBounds[1]
    xmin, xmax = box_min[0], box_max[0]
    ymin, ymax = box_min[1], box_max[1]
    zmin, zmax = box_min[2], box_max[2]

    box_height = ymax-ymin
    strain_velocity = params.strain_rate * box_height * 1e-12

    # Now we set them to be lammps variables so they are still accessible later on
    lmp.command(f"variable ramp_timestep equal (step-{params.thermo_time})") # Create ramp timestep

    lmp.command(f"variable ramp_velocity_top equal (({strain_velocity}*v_ramp_timestep)/{params.ramp_time})") #
    lmp.command(f"variable shear_velocity_top equal {strain_velocity}") # 

    return None

def ramp(lmp, params, paths):

    lmp.command(f"fix 2 fixgrp setforce 0.0 0.0 0.0")

    lmp.command("velocity topgrp set v_ramp_velocity_top 0.0 0.0")
    lmp.command("velocity botgrp set 0.0 0.0 0.0")

    lmp.command(f"run {params.ramp_time}")

    return None

def shear(lmp, params, paths):

    lmp.command("velocity topgrp set v_shear_velocity_top 0.0 0.0")
    lmp.command("velocity botgrp set 0.0 0.0 0.0")

    lmp.command(f"run {params.run_time}")