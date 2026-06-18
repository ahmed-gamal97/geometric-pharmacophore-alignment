# Geometric Pharmacophore Alignment

A cross-docking solution that places small molecules into protein binding pockets defined only by pharmacophore interaction sites and exclusion spheres — no explicit protein structure required.

## Problem Summary

Given 5 drug-like molecules (as SMILES strings), each paired with:
- **Interaction sites**: 3D positions where the ligand should make contact, typed by chemical family (Donor / Acceptor / Hydrophobe / Aromatic)
- **Exclusion volumes**: Steric spheres the ligand must not penetrate

The goal is to find the best 3D pose for each molecule that **maximises the pharmacophore alignment score** while **avoiding all steric clashes**.

Score formula:
```
score = Σ w_i * exp(-(d_i / 1.25)²)
```
where `d_i` = minimum distance from site `i` to the nearest ligand atom whose chemical feature matches the site's family.

## Approach

### 1. 3D Conformer Generation
Generate 200 low-energy 3D conformers per molecule using RDKit's ETKDGv3 method (experimentally-informed torsion preferences) followed by MMFF force-field minimisation.

### 2. Pharmacophore Feature Assignment
Use RDKit's `BaseFeatures.fdef` factory to label each heavy atom with one or more pharmacophore families. Mapping:

| RDKit family | Task family |
|---|---|
| Donor | Donor |
| Acceptor | Acceptor |
| Hydrophobe, LumpedHydrophobe | Hydrophobe |
| Aromatic | Aromatic |

### 3. Rigid-Body Pose Search

**Phase 1 — Anchor-based search**
For each (interaction site, matching ligand atom) pair:
- Translate the molecule so that atom lands exactly on the site
- Try 40 random orientations (uniform SO(3) sampling)
- Check all atoms including H against exclusion volumes
- This guarantees at least one atom-site match per trial

**Phase 2 — Global random search**
Try 200 uniform-random SO(3) rotations with translation near the pharmacophore centroid, checking all atoms for clashes.

**Phase 3 — Global top-K refinement**
Collect top-10 candidates per conformer (up to 2000 total), sort globally, refine the best 60 with Nelder-Mead:
- 7-parameter optimisation: quaternion (4) + translation (3)
- Smooth squared-violation clash penalty guides the optimiser away from forbidden regions
- Clash check (all atoms including H) after each refinement

### 4. Selection
The highest-scoring clash-free pose across all conformers and all rotations is written to the output SDF.

## Running

### With Docker (matches grading environment exactly)

```bash
# Build the image
docker build -t pharma-align .

# Run (mount your data directory)
docker run --rm \
  -v /path/to/data:/root/data:ro \
  -v /path/to/results:/root/results \
  pharma-align
```

The container reads `/root/data/targets.json` and writes `/root/results/docked_poses.sdf`.

### Locally (if RDKit + scipy are installed)

```bash
pip install rdkit scipy numpy
python solve.py
```

By default `solve.py` reads from `/root/data/targets.json`. Edit the `data_path` and `output_path` variables at the bottom of the file to use local paths.

## Dependencies

| Package | Version |
|---|---|
| rdkit | ≥ 2023.3.1 |
| scipy | ≥ 1.11.0 |
| numpy | ≥ 1.24.0 |

## Output

A single SDF file `docked_poses.sdf` containing one 3D conformer per target, in the same order as `targets.json`. Each molecule retains the original SMILES atom count and topology (no explicit hydrogens added). The `Score` property in the SDF stores the final alignment score.

## Results

All 5 poses are clash-free (no atom within 1.1 Å of any exclusion centre).

| Target | Molecule | Score | Max Score | % |
|--------|----------|-------|-----------|---|
| target_1 | Ibuprofen | 4.807 | 5.400 | **89.0%** |
| target_2 | Caffeine | 3.576 | 7.100 | 50.4% |
| target_3 | Aspirin | 5.601 | 8.300 | **67.5%** |
| target_4 | Imatinib-like (37 atoms) | 7.702 | 12.600 | **61.1%** |
| target_5 | Gefitinib-like (31 atoms) | 5.893 | 10.750 | 54.8% |

**Note on target_2 (Caffeine)**: Caffeine has no N-H or O-H bonds so it has zero Donor atoms. Its three methyl groups are N-methyls (directly bonded to nitrogen) so RDKit's feature factory does not classify them as Hydrophobe. The pocket has 1 Donor site (weight 1.4) and 1 Hydrophobe site (weight 0.7) that caffeine structurally cannot satisfy — this is a property of the molecule, not the alignment algorithm. The achievable score for caffeine is 72% of the pharmacophore-accessible maximum (3.58 / 5.00).
