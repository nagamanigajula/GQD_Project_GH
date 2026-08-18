# -*- coding: utf-8 -*-
"""
Fusion-based hexagonal lattice growth

Generates graphene quantum dot (GQD) structures by starting from a single
hexagon and growing outward one ring at a time, up to a random target size
between 20 and 40 rings. Each structure is checked for validity (sanitizable
by RDKit, correct ring count, sp2 fraction, synthetic accessibility score,
and uniqueness) before being accepted.

Usage:
  python 01_structure_generation.py --target 100000

Resuming:
  Re-running the same command will pick up from the existing output CSV
  instead of starting over.
"""

import os, sys, math, random, time, warnings, argparse
from collections import defaultdict

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors, QED
from rdkit.Chem.rdchem import HybridizationType

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fusion-based hexagonal lattice growth generation")
parser.add_argument("--target", type=int, default=100_000,
                     help="Number of unique structures to generate (default: 100,000)")
parser.add_argument("--min_hex", type=int, default=20, help="Minimum hexagonal rings (default: 20)")
parser.add_argument("--max_hex", type=int, default=40, help="Maximum hexagonal rings (default: 40)")
parser.add_argument("--max_sa", type=float, default=8.0, help="SA score cutoff (default: 8.0)")
parser.add_argument("--min_sp2", type=float, default=0.95, help="Minimum sp2 fraction (default: 0.95)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
parser.add_argument("--checkpoint", type=int, default=1000,
                     help="Save a CSV checkpoint every N accepted structures (default: 1000)")
parser.add_argument("--max_attempts_mult", type=int, default=30,
                     help="Attempt budget = target * this multiplier (default: 30)")
parser.add_argument("--save_images", action="store_true",
                     help="Save a PNG for each accepted structure (off by default)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Synthetic accessibility scorer
# ---------------------------------------------------------------------------
try:
    import sascorer
    HAS_SA = True
except ImportError:
    HAS_SA = False
    print("WARNING: sascorer.py not found - SA score filtering will be skipped.")
    print("  Fix: wget -q https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/sascorer.py")
    print("       wget -q https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz\n")

# ---------------------------------------------------------------------------
# Config and output folders
# ---------------------------------------------------------------------------
TARGET      = args.target
MIN_HEX     = args.min_hex
MAX_HEX     = args.max_hex
MAX_SA      = args.max_sa
MIN_SP2     = args.min_sp2
RANDOM_SEED = args.seed
CHECKPOINT  = args.checkpoint
MAX_ATTEMPTS = TARGET * args.max_attempts_mult

OUT_CSV    = "hex_lattice_structures.csv"
OUT_MOLS   = "mol_files"
OUT_IMAGES = "structure_images"

os.makedirs(OUT_MOLS, exist_ok=True)
if args.save_images:
    os.makedirs(OUT_IMAGES, exist_ok=True)

rng = random.Random(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Hexagonal grid growth
# ---------------------------------------------------------------------------
HEX_DIR = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]

def hex_neighbors(q, r):
    return [(q+dq, r+dr) for dq, dr in HEX_DIR]

def grow_cluster(n):
    """Grow a cluster of n hexagons starting from a single seed hexagon at (0,0)."""
    cluster  = {(0, 0)}
    frontier = list(hex_neighbors(0, 0))
    while len(cluster) < n and frontier:
        rng.shuffle(frontier)
        for i, c in enumerate(frontier):
            if any(c in hex_neighbors(*h) for h in cluster):
                cluster.add(c)
                frontier.pop(i)
                for nb in hex_neighbors(*c):
                    if nb not in cluster and nb not in frontier:
                        frontier.append(nb)
                break
    return cluster

# Six ways to build the 3-atom corner of a hexagon at (q, r), used to figure
# out which corners are shared between neighboring hexagons.
VERT_FNS = [
    lambda q, r: frozenset([(q, r), (q+1, r-1), (q+1, r)]),
    lambda q, r: frozenset([(q, r), (q+1, r), (q, r+1)]),
    lambda q, r: frozenset([(q, r), (q, r+1), (q-1, r+1)]),
    lambda q, r: frozenset([(q, r), (q-1, r+1), (q-1, r)]),
    lambda q, r: frozenset([(q, r), (q-1, r), (q, r-1)]),
    lambda q, r: frozenset([(q, r), (q, r-1), (q+1, r-1)]),
]

def cluster_to_mol(cluster):
    """Convert a set of hexagon coordinates into an RDKit molecule."""
    vert_to_idx = {}
    hex_map = {}
    for (q, r) in cluster:
        atoms = []
        for fn in VERT_FNS:
            key = fn(q, r)
            if key not in vert_to_idx:
                vert_to_idx[key] = len(vert_to_idx)
            atoms.append(vert_to_idx[key])
        hex_map[(q, r)] = atoms

    n_atoms = len(vert_to_idx)

    def center(q, r):
        return math.sqrt(3) * (q + r / 2.0), 1.5 * r

    # Average the three hexagon centers meeting at each shared corner to get
    # that corner's 2D position.
    pos = {}
    for key, idx in vert_to_idx.items():
        xs, ys = [], []
        for h in key:
            x, y = center(h[0], h[1])
            xs.append(x); ys.append(y)
        pos[idx] = (sum(xs) / 3, sum(ys) / 3)

    rw = Chem.RWMol()
    for _ in range(n_atoms):
        rw.AddAtom(Chem.Atom(6))

    bonds = set()
    for atoms in hex_map.values():
        for i in range(6):
            a1, a2 = atoms[i], atoms[(i+1) % 6]
            b = (min(a1, a2), max(a1, a2))
            if b not in bonds:
                bonds.add(b)
                rw.AddBond(a1, a2, Chem.BondType.AROMATIC)

    # Scale coordinates using the standard aromatic C-C bond length (1.42 A)
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        x, y = pos[i]
        conf.SetAtomPosition(i, (x * 1.42, y * 1.42, 0))
    rw.AddConformer(conf)

    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol, canonical=True)
        if '.' in smi:
            # more than one disconnected piece - not a single structure
            return None, None
        AllChem.Compute2DCoords(mol)
        return mol, smi
    except Exception:
        return None, None

# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------
def get_sp2(mol):
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    return sum(a.GetHybridization() == HybridizationType.SP2 for a in heavy) / len(heavy)

def get_sa(mol):
    if not HAS_SA:
        return None
    try:
        return round(sascorer.calculateScore(mol), 3)
    except Exception:
        return 999

def get_inchi(mol):
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None

def count_hex(mol):
    ri = mol.GetRingInfo()
    return sum(1 for r in ri.AtomRings() if len(r) == 6)

def compute_descriptors(mol):
    row = {}
    try: row["Molecular_Weight"] = round(Descriptors.MolWt(mol), 2)
    except Exception: row["Molecular_Weight"] = "NA"
    try: row["LogP"] = round(Descriptors.MolLogP(mol), 2)
    except Exception: row["LogP"] = "NA"
    try: row["TPSA"] = round(rdMolDescriptors.CalcTPSA(mol), 2)
    except Exception: row["TPSA"] = "NA"
    try: row["HBD"] = rdMolDescriptors.CalcNumHBD(mol)
    except Exception: row["HBD"] = "NA"
    try: row["HBA"] = rdMolDescriptors.CalcNumHBA(mol)
    except Exception: row["HBA"] = "NA"
    try: row["Rotatable_Bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    except Exception: row["Rotatable_Bonds"] = "NA"
    try:
        mol_qed = Chem.Mol(mol)
        Chem.Kekulize(mol_qed, clearAromaticFlags=True)
        Chem.SanitizeMol(mol_qed)
        row["QED"] = round(QED.qed(mol_qed), 3)
    except Exception:
        row["QED"] = "NA"
    return row

# ===========================================================================
# Main generation loop
# ===========================================================================
print("\n" + "="*65)
print(f"  Fusion-based hexagonal lattice growth generation")
print(f"  Target: {TARGET:,} unique structures")
print("="*65 + "\n")

records = []
seen_inchi = set()
stats = defaultdict(int)
struct_id = 0
attempt = 0
t_start = time.time()

if os.path.exists(OUT_CSV):
    try:
        df_prev = pd.read_csv(OUT_CSV)
        if len(df_prev) > 0 and "InChIKey" in df_prev.columns:
            records = df_prev.to_dict("records")
            seen_inchi = set(df_prev["InChIKey"].dropna().tolist())
            struct_id = len(records)
            print(f"Resuming: found {struct_id:,} structures already generated "
                  f"in '{OUT_CSV}' - continuing from there.\n")
    except Exception as e:
        print(f"Could not read existing '{OUT_CSV}' ({e}); starting fresh.\n")

while struct_id < TARGET and attempt < MAX_ATTEMPTS:
    attempt += 1

    cluster = grow_cluster(rng.randint(MIN_HEX, MAX_HEX))
    mol, smi = cluster_to_mol(cluster)
    if mol is None:
        stats["invalid_structure"] += 1
        continue

    key = get_inchi(mol)
    if key is None or key in seen_inchi:
        stats["duplicate"] += 1
        continue

    h = count_hex(mol)
    if not (MIN_HEX <= h <= MAX_HEX):
        stats["hex_fail"] += 1
        continue

    sp2 = get_sp2(mol)
    if sp2 < MIN_SP2:
        stats["sp2_fail"] += 1
        continue

    sa = get_sa(mol)
    if sa is not None and sa > MAX_SA:
        stats["sa_fail"] += 1
        continue

    # structure passed all checks - accept it
    seen_inchi.add(key)
    struct_id += 1
    name = f"gqd_{struct_id:06d}"

    desc = compute_descriptors(mol)
    desc.update({
        "Structure_Name": name,
        "SMILES": smi,
        "InChIKey": key,
        "N_Hex": h,
        "SP2": round(sp2, 3),
        "SA_Score": sa,
    })
    records.append(desc)

    try:
        Chem.MolToMolFile(mol, os.path.join(OUT_MOLS, f"{name}.mol"))
    except Exception:
        pass

    if args.save_images:
        try:
            Draw.MolToFile(mol, os.path.join(OUT_IMAGES, f"{name}.png"), size=(600, 600))
        except Exception:
            pass

    if struct_id % CHECKPOINT == 0:
        elapsed = time.time() - t_start
        rate = struct_id / elapsed
        eta_min = (TARGET - struct_id) / rate / 60 if rate > 0 else float("inf")
        pd.DataFrame(records).to_csv(OUT_CSV, index=False)
        print(f"Progress: {struct_id:>7,}/{TARGET:,}  "
              f"attempts={attempt:,}  rate={rate:.1f}/s  "
              f"ETA~{eta_min:.1f} min  [checkpoint saved]")

# Save final results
df_out = pd.DataFrame(records)
df_out.to_csv(OUT_CSV, index=False)

elapsed_total = time.time() - t_start
print(f"\n{'='*65}")
print(f"  GENERATION COMPLETE")
print(f"{'='*65}")
print(f"  Structures generated : {len(df_out):,}")
print(f"  Total attempts       : {attempt:,}")
print(f"  Total time           : {elapsed_total/60:.1f} minutes")
print(f"\n  Rejection stats:")
for k, v in stats.items():
    print(f"    {k:<20}: {v:,}")
if len(df_out):
    print(f"\n  Hex ring range : {df_out['N_Hex'].min()} - {df_out['N_Hex'].max()}")
    print(f"  Avg SP2        : {df_out['SP2'].mean():.4f}")
print(f"\n  Saved -> {OUT_CSV}")
print(f"  MOL files -> {OUT_MOLS}/")
