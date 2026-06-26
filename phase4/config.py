"""
config.py — Phase 4 configuration: which datasets, which devices, which knobs.

Everything that is environment-specific (paths) or a measurement convention
(thresholds, normalizer) lives here so the engine modules stay generic.

IMPORTANT — camera/device labels are DEFAULTS, verify against each dataset's
own documentation before quoting them in the report. The cross-dataset story
hinges on attributing drops to the acquisition device (the MuCaRD framing), so
the device labels must be right. Notes below record the commonly-cited setups;
treat them as a starting point, not ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The single in-domain dataset all degradation is measured against.
IN_DOMAIN_DATASET = "REFUGE2"

# Optic-disc diameter used to normalize the fovea localization error.
# "vertical" matches the vertical-CDR convention; swap to "equivalent" if you
# prefer an area-based normalizer.
LOC_NORM_DIAMETER = "vertical"

# A localization is "successful" if it lands within this many disc diameters.
LOC_SUCCESS_DISC_FRACTION = 1.0

# Mask threshold for turning probability maps into binary masks.
MASK_THRESHOLD = 0.5

# Failure-analysis heuristics (per-image tags; see failure_analysis.py).
FAIL_DISC_DICE = 0.20        # disc essentially missed
FAIL_CUP_DICE = 0.20         # cup essentially missed (only when cup GT exists)
FAIL_CUP_OVERSEG_RATIO = 2.0  # pred cup area > 2x GT cup area
FAIL_LOC_NORM = 1.0          # fovea off by more than one disc diameter
FAIL_CLS_CONF = 0.40         # |prob-0.5| > 0.40 => "confident"; wrong+confident = bad

# How many worst cases to export per task in failure analysis.
N_FAILURE_CASES = 12


@dataclass
class DatasetSpec:
    name: str
    root: str                       # filesystem root (used only in --mode real)
    device_default: str             # camera label if the set is single-device
    has_cup: bool                   # cup annotated?
    has_fovea: bool                 # fovea annotated?
    has_disc: bool = True
    has_cls: bool = True            # glaucoma label present?
    multi_device: bool = False      # per-image device set by the adapter
    split: str = "test"
    notes: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Dataset registry. Roots are placeholders — point them at your Phase 1 layout.
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, DatasetSpec] = {
    # In-domain core. NOTE: REFUGE itself bakes in a camera shift — the training
    # split was captured on a Zeiss Visucam 500 while val/test came from a Canon
    # CR-2. So even "in-domain test" already lives on a different device than
    # training; the device column makes that explicit rather than hiding it.
    "REFUGE2": DatasetSpec(
        name="REFUGE2", root="data/refuge2/test",
        device_default="Canon_CR2", has_cup=True, has_fovea=True,
        notes="In-domain test. Train=Zeiss Visucam 500, Val/Test=Canon CR-2.",
    ),
    # 6 independent expert annotators -> also feeds inter-rater variability.
    "RIGA": DatasetSpec(
        name="RIGA", root="data/riga",
        device_default="mixed", has_cup=True, has_fovea=False,
        multi_device=True,
        notes="Subsets BinRushed / Magrabia / MESSIDOR captured on different "
              "cameras; adapter sets device per image from the subset folder.",
    ),
    # Same disc structure, different disease (pathologic myopia): transfer test.
    "PALM": DatasetSpec(
        name="PALM", root="data/palm",
        device_default="Zeiss_Visucam", has_cup=False, has_fovea=True,
        has_cls=True,
        notes="Pathologic myopia. Disc + fovea annotated; NO cup -> cup Dice "
              "and CDR are not scored here (availability flags handle it).",
    ),
    # External segmentation sets from yet other cameras (pick 1-2).
    "Drishti-GS": DatasetSpec(
        name="Drishti-GS", root="data/drishti_gs",
        device_default="Drishti_fundus", has_cup=True, has_fovea=False,
        notes="Aravind Eye Hospital fundus camera; disc+cup (multi-expert cup).",
    ),
    "RIM-ONE": DatasetSpec(
        name="RIM-ONE", root="data/rim_one_dl",
        device_default="Nidek_AFC210", has_cup=True, has_fovea=False,
        notes="RIM-ONE DL release; disc+cup.",
    ),
    "G1020": DatasetSpec(
        name="G1020", root="data/g1020",
        device_default="G1020_camera", has_cup=True, has_fovea=False,
        notes="1020 fundus images with disc/cup masks + glaucoma labels.",
    ),
}

# Default external sets to evaluate (besides the in-domain one).
DEFAULT_EXTERNAL = ["RIGA", "PALM", "Drishti-GS"]

# Devices that count as "in-domain" for the device-stratified view. Everything
# else is treated as a shifted device when computing per-device degradation.
IN_DOMAIN_DEVICES = {"Canon_CR2", "Zeiss_Visucam"}
