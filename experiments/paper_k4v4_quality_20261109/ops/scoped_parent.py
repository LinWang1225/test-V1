"""User-requested scheduling override; numerical sources remain frozen."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'staged_vllm_recovery_20261106'
sys.path.insert(0,str(PARENT/'src'))
import run_queue
from provenance import freeze,QUEUE
from common import atomic

frozen=freeze()
selected=tuple(c for c in QUEUE if c!='U_Clipped')
atomic(PARENT/'queue_control.json',dict(
    user_requested=True,cancelled_methods=['U_Clipped'],selected_methods=list(selected),
    numeric_fingerprint_unchanged=frozen['fingerprint'],
    reason='User cancelled U_Clipped and prioritized paper TurboQuant and author-official KVarN'))
# Only the launcher loop changes. The numerical provenance module still
# verifies the original full source/input/codec manifest byte for byte.
run_queue.QUEUE=selected
run_queue.main()
