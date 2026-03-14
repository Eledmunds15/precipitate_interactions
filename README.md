# precipitate_interactions
Simulating dislocation climb around precipitates with LAMMPS

```
python simulations/01_shear/run.py --temperature 800 --strain_rate 1e7 --input /home/Ethan/Projects/prec_interactions/input_TEST/Fe_E111_110_R20.lmp --potential /home/Ethan/Projects/prec_interactions/potentials/mendelev03.fs
```

```
mpirun -np 24 python simulations/01_shear/run.py --temperature 800 --strain_rate 1e7 --input /home/Ethan/Projects/prec_interactions/input_TEST/Fe_E111_110_R20.lmp --potential /home/Ethan/Projects/prec_interactions/potentials/mendelev03.fs
```
