#!/usr/bin/env python3
"""
Geometric Pharmacophore Alignment
==================================
Cross-docking without an explicit protein structure:
  - Generate 3D conformers from SMILES
  - Assign pharmacophore features (Donor/Acceptor/Hydrophobe/Aromatic) to atoms
  - Search for the rigid-body pose that maximises the Gaussian-decay score
      score = Σ w_i * exp(-(d_i / 1.25)^2)
    where d_i = distance from site i to its nearest matching-family ligand atom
  - Reject poses where ANY atom (including H) clashes with an exclusion sphere
    (clash = atom within 1.2 Å of exclusion centre; 0.1 Å tolerance → 1.1 Å)
  - Write the single best pose per target (original SMILES atom count) to SDF
"""

import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures

# ---------------------------------------------------------------------------
# Feature factory (loaded once at module level)
# ---------------------------------------------------------------------------
_FDEF_PATH = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_FACTORY   = ChemicalFeatures.BuildFeatureFactory(_FDEF_PATH)

_RDKIT_TO_TASK = {
    "Donor":          "Donor",
    "Acceptor":       "Acceptor",
    "Hydrophobe":     "Hydrophobe",
    "LumpedHydrophobe": "Hydrophobe",
    "Aromatic":       "Aromatic",
}

CLASH_RADIUS    = 1.2          # Å exclusion sphere radius (spec)
CLASH_TOLERANCE = 0.1          # Å tolerance (spec)
CLASH_THRESHOLD = CLASH_RADIUS - CLASH_TOLERANCE   # 1.1 Å
PENALTY_WEIGHT  = 500.0        # weight for clash penalty in Nelder-Mead


# ---------------------------------------------------------------------------
# Feature assignment (heavy atoms only — features never sit on H)
# ---------------------------------------------------------------------------

def assign_atom_families(mol):
    """Returns dict: task_family -> sorted list of heavy-atom indices in mol."""
    result = {f: set() for f in ("Donor", "Acceptor", "Hydrophobe", "Aromatic")}
    for feat in _FACTORY.GetFeaturesForMol(mol):
        task_fam = _RDKIT_TO_TASK.get(feat.GetFamily())
        if task_fam is None:
            continue
        for idx in feat.GetAtomIds():
            if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1:
                result[task_fam].add(idx)
    return {k: sorted(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Scoring  (heavy atoms only — pharmacophore features are on heavy atoms)
# ---------------------------------------------------------------------------

def score_pose(coords_heavy, family_idx, sites):
    """
    score = Σ_i  w_i * exp(-(d_i / 1.25)^2)
    d_i = min distance from site i to nearest matching heavy atom.

    coords_heavy : (n_heavy, 3) — heavy-atom positions
    family_idx   : dict  task_family -> [local indices into coords_heavy]
    """
    total = 0.0
    for site in sites:
        idxs = family_idx.get(site["family"])
        if not idxs:
            continue
        sp   = np.array([site["x"], site["y"], site["z"]], dtype=np.float64)
        diff = coords_heavy[idxs] - sp
        d    = np.sqrt((diff * diff).sum(axis=1)).min()
        total += site["weight"] * np.exp(-(d / 1.25) ** 2)
    return total


# ---------------------------------------------------------------------------
# Clash checking — ALL atoms including hydrogens (spec: "any ligand atom")
# ---------------------------------------------------------------------------

def clash_penalty_smooth(coords_all, excluded_volumes):
    """
    Smooth squared-violation penalty for Nelder-Mead guidance.
    Uses CLASH_THRESHOLD (1.1 Å) so the penalty is zero for all poses
    that pass the final is_clash_free() check — the optimizer is never
    pushed away from valid space.
    """
    pen = 0.0
    for ev in excluded_volumes:
        c    = np.array([ev["x"], ev["y"], ev["z"]], dtype=np.float64)
        diff = coords_all - c
        dist = np.sqrt((diff * diff).sum(axis=1))
        viol = np.maximum(0.0, CLASH_THRESHOLD - dist)   # zero for d >= 1.1 Å
        pen += (viol * viol).sum()
    return pen


def is_clash_free(coords_all, excluded_volumes):
    """True when no atom (including H) is within CLASH_THRESHOLD of any centre."""
    for ev in excluded_volumes:
        c    = np.array([ev["x"], ev["y"], ev["z"]], dtype=np.float64)
        diff = coords_all - c
        if np.sqrt((diff * diff).sum(axis=1)).min() < CLASH_THRESHOLD:
            return False
    return True


# ---------------------------------------------------------------------------
# Rigid-body transform — quaternion parameterisation (no trig in hot loop)
# ---------------------------------------------------------------------------

def quat_to_matrix(q):
    """Unnormalized [w,x,y,z] quaternion → 3×3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def apply_quat(centered, q, t):
    """Rotate then translate: (N,3) → (N,3)."""
    return centered @ quat_to_matrix(q).T + t


def rotvec_to_quat(rv):
    xyzw = Rotation.from_rotvec(rv).as_quat()   # scipy gives [x,y,z,w]
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])   # → [w,x,y,z]


# ---------------------------------------------------------------------------
# Candidate collection (per conformer, no refinement)
#
# We keep TWO centred arrays per conformer:
#   centered_h   — heavy atoms only  (for anchor selection and scoring)
#   centered_all — all atoms incl. H (for clash checking)
# Both are centred by the heavy-atom centroid so the same (R,t) applies.
# ---------------------------------------------------------------------------

def collect_candidates(centered_h, centered_all, ha_arr,
                       family_idx, sites, excluded_volumes, rng,
                       n_anchor_rots=40, n_random_rots=200, top_k=10):
    """
    Phase 1 (anchor): place each matching heavy atom on each matching site,
    try n_anchor_rots random orientations, check ALL atoms for clashes.

    Phase 2 (random): n_random_rots uniform-SO(3) rotations near the centroid.

    Returns list of (score, rotvec, t, centered_h, centered_all).
    """
    if not sites:
        return []
    site_centroid = np.mean([[s["x"], s["y"], s["z"]] for s in sites], axis=0)
    candidates    = []

    anc_rots = Rotation.random(n_anchor_rots, random_state=int(rng.integers(1 << 31)))
    for site in sites:
        sp = np.array([site["x"], site["y"], site["z"]], dtype=np.float64)
        for atom_idx in family_idx.get(site["family"], []):
            a_cen = centered_h[atom_idx]
            for rot in anc_rots:
                t          = sp - rot.apply(a_cen)
                coords_all = rot.apply(centered_all) + t    # ALL atoms
                if is_clash_free(coords_all, excluded_volumes):
                    coords_h = coords_all[ha_arr]            # heavy atoms only
                    s = score_pose(coords_h, family_idx, sites)
                    candidates.append((s, rot.as_rotvec(), t))

    glob_rots = Rotation.random(n_random_rots, random_state=int(rng.integers(1 << 31)))
    for rot in glob_rots:
        t          = site_centroid + rng.standard_normal(3) * 2.5
        coords_all = rot.apply(centered_all) + t
        if is_clash_free(coords_all, excluded_volumes):
            coords_h = coords_all[ha_arr]
            s = score_pose(coords_h, family_idx, sites)
            candidates.append((s, rot.as_rotvec(), t))

    if not candidates:
        return []

    candidates.sort(key=lambda x: -x[0])
    return [(s, rv, t, centered_h, centered_all) for s, rv, t in candidates[:top_k]]


# ---------------------------------------------------------------------------
# Local refinement with clash penalty so Nelder-Mead stays in valid space
# ---------------------------------------------------------------------------

def refine_candidate(rv0, t0, centered_h, centered_all, ha_arr,
                     family_idx, sites, excluded_volumes):
    """
    Minimise -score subject to a smooth clash penalty that steers the
    optimiser away from excluded volumes (all atoms, including H).

    Returns (heavy_coords, score) or (None, -inf) if the refined pose clashes.
    """
    q0 = rotvec_to_quat(rv0)
    x0 = np.concatenate([q0, t0])

    def objective(params):
        q, t   = params[:4], params[4:]
        ca     = apply_quat(centered_all, q, t)
        pen    = clash_penalty_smooth(ca, excluded_volumes)
        if pen > 0.0:
            # Guide optimiser away; smooth so Nelder-Mead can follow the slope
            return PENALTY_WEIGHT * pen
        ch = ca[ha_arr]
        return -score_pose(ch, family_idx, sites)

    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-4,
                            "fatol": 1e-5, "adaptive": True})

    ca = apply_quat(centered_all, res.x[:4], res.x[4:])
    if is_clash_free(ca, excluded_volumes):
        ch = ca[ha_arr]
        return ch, score_pose(ch, family_idx, sites)
    return None, float("-inf")


# ---------------------------------------------------------------------------
# Per-target pipeline
# ---------------------------------------------------------------------------

def process_target(name, target):
    smiles   = target["smiles"]
    sites    = target["interaction_sites"]
    excluded = target["excluded_volumes"]

    print(f"\n[{name}]  {smiles}")
    print(f"  {len(sites)} sites | {len(excluded)} exclusion volumes", flush=True)

    mol   = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol_h  = Chem.AddHs(mol)
    n_orig = mol.GetNumAtoms()   # heavy-atom count = original SMILES count

    # ha_map: mol_h atom indices that are heavy atoms (ordered)
    ha_map = [i for i in range(mol_h.GetNumAtoms())
              if mol_h.GetAtomWithIdx(i).GetAtomicNum() != 1]
    ha_arr = np.array(ha_map, dtype=int)     # numpy for fast indexing
    h2l    = {v: k for k, v in enumerate(ha_map)}   # mol_h idx → local 0-based

    # Pharmacophore features in local (0..n_heavy-1) indexing
    full_feats  = assign_atom_families(mol_h)
    local_feats = {fam: [h2l[i] for i in idxs if i in h2l]
                   for fam, idxs in full_feats.items()}
    print(f"  Features: { {k: len(v) for k, v in local_feats.items()} }")

    # Generate 200 low-energy conformers (with H for better geometry)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 0
    AllChem.EmbedMultipleConfs(mol_h, numConfs=200, params=params)
    if mol_h.GetNumConformers() == 0:
        AllChem.EmbedMultipleConfs(mol_h, numConfs=50, params=AllChem.ETKDG())
    AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=0)
    n_confs = mol_h.GetNumConformers()
    print(f"  {n_confs} conformers generated", flush=True)

    # Collect top-10 candidates per conformer (both heavy and full coords)
    rng = np.random.default_rng(seed=42)
    global_cands = []   # (score, rv, t, centered_h, centered_all)

    for cid in range(n_confs):
        all_pos  = np.array(mol_h.GetConformer(cid).GetPositions())
        heavy_pos = all_pos[ha_arr]
        centroid  = heavy_pos.mean(axis=0)
        c_h   = heavy_pos - centroid          # (n_heavy, 3) centred heavy atoms
        c_all = all_pos   - centroid          # (n_all,   3) centred all atoms

        cands = collect_candidates(c_h, c_all, ha_arr,
                                   local_feats, sites, excluded, rng)
        global_cands.extend(cands)

    # Global sort → refine top 60
    if not global_cands:
        # No clash-free candidate at all — try many random translations
        print("  Warning: no clash-free candidate; running translation search.")
        sc      = np.mean([[s["x"], s["y"], s["z"]] for s in sites], axis=0)
        all_pos = np.array(mol_h.GetConformer(0).GetPositions())
        hp      = all_pos[ha_arr]
        centroid = hp.mean(axis=0)
        c_h_0   = hp       - centroid
        c_all_0 = all_pos  - centroid

        best_score  = float("-inf")
        best_coords = None
        for _ in range(2000):
            t = sc + rng.standard_normal(3) * 3.0
            ca = c_all_0 + t
            if is_clash_free(ca, excluded):
                ch = ca[ha_arr]
                s  = score_pose(ch, local_feats, sites)
                if s > best_score:
                    best_score  = s
                    best_coords = ch.copy()

        if best_coords is None:
            print("  ERROR: could not find any clash-free pose — skipping target.")
            return None, float("-inf")
    else:
        global_cands.sort(key=lambda x: -x[0])
        n_ref = min(60, len(global_cands))
        print(f"  {len(global_cands)} candidates → refining top {n_ref}", flush=True)

        best_score  = float("-inf")
        best_coords = None

        for sc0, rv0, t0, c_h, c_all in global_cands[:n_ref]:
            ch, s = refine_candidate(rv0, t0, c_h, c_all, ha_arr,
                                     local_feats, sites, excluded)
            if s > best_score:
                best_score  = s
                best_coords = ch

        if best_coords is None:
            # All Nelder-Mead refinements ended in clashes; fall back to the best
            # raw (unrefined) candidate — it was already verified clash-free during
            # collection, so this is always a valid pose.
            sc0, rv0, t0, c_h, c_all = global_cands[0]
            ca = Rotation.from_rotvec(rv0).apply(c_all) + t0
            best_coords = ca[ha_arr]
            best_score  = sc0

    max_score = sum(s["weight"] for s in sites)
    pct = 100.0 * best_score / max_score if max_score > 0 else 0.0
    print(f"  Best score: {best_score:.4f} / {max_score:.4f}  ({pct:.1f}%)")

    # Build output molecule: original atom count (no explicit H), best coordinates
    conf = Chem.Conformer(n_orig)
    for j, xyz in enumerate(best_coords):
        conf.SetAtomPosition(j, xyz.tolist())
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol, best_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    data_path   = "/root/data/targets.json"
    output_path = "/root/results/docked_poses.sdf"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(data_path) as fh:
        targets = json.load(fh)

    print(f"Loaded {len(targets)} targets")
    writer = Chem.SDWriter(output_path)

    for name, target in targets.items():
        mol, score = process_target(name, target)
        if mol is None:
            continue   # no clash-free pose found; spec forbids outputting it
        mol.SetProp("_Name", name)
        mol.SetProp("Score", f"{score:.6f}")
        writer.write(mol)

    writer.close()
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
