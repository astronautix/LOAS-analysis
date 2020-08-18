# 4
Résultats de single_axis avec loas < 0.4.0 (vent selon z)

(Comme 3 mais avec 1e5 particules au lieu de 1e7)

Avec `w` dans `np.linspace(-200,200,101)`. Chaque fichier est la sortie de `drag.runSim` pour la valeur de `w` correspondant au nom du fichier.
```python
mesh = trimesh.load_mesh("../models/crocus/45.stl")
drag = loas.rad.RAD(
    sat_mesh = mesh,
    model = loas.rad.models.maxwell(0.10),
    part_per_iteration = 1e7,
    nb_workers = 8
)
drag.start()
sat_Q = [loas.utils.Quaternion(math.cos(angle/2), math.sin(angle/2), 0, 0) for angle in np.linspace(0, 2*math.pi, 10)]
res = drag.runSim(
    sat_W = loas.utils.tov(float(w),0,0),
    sat_Q = sat_Q,
    sat_speed = 7000,
    sat_temp = 300,
    part_density = 1e-11,
    part_mol_mass = 0.016,
    part_temp = 1800,
    with_drag_coef = True
)
```
