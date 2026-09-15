"""Central configuration. Change things here, not in the scripts."""
from pathlib import Path

# ---- model ----------------------------------------------------------------
# Qwen is ungated on HuggingFace -- no access request needed.
# Swap to "Qwen/Qwen2.5-3B-Instruct" later to test whether the signal
# gets stronger with scale.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOAD_IN_4BIT = True

# ---- data -----------------------------------------------------------------
DATASET = "mandarjoshi/trivia_qa"
DATASET_CONFIG = "rc.nocontext"
SPLIT = "validation"
N_EXAMPLES = 2000          # start here; raise once the pipeline works
MAX_NEW_TOKENS = 20

# ---- extraction -----------------------------------------------------------
SHARD_SIZE = 250           # examples per saved shard (disconnect insurance)
POOLING = "mean"           # "mean" over answer tokens, or "last"

# ---- probe ----------------------------------------------------------------
TEST_FRACTION = 0.3
RANDOM_SEED = 0

# ---- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
for _d in (DATA, ARTIFACTS, FIGURES):
    _d.mkdir(exist_ok=True)

ANSWERS_FILE = DATA / "answers.jsonl"
ACTS_PREFIX = DATA / "acts"        # -> acts_00000.npy, acts_00001.npy, ...
LABELS_FILE = DATA / "labels.npy"
GROUPS_FILE = DATA / "groups.npy"
