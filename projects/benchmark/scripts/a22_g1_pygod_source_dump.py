"""A22-G1 helper: dump installed PyGOD source hashes and the training-loop source actually used."""
import hashlib
import importlib.metadata as m
import inspect
import json
import sys

import pygod
import pygod.detector.base as base
import pygod.detector.dominant as det_dom
import pygod.nn.dominant as nn_dom
from pygod.detector import DOMINANT
from pygod.nn import functional as F


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


info = {
    "pygod_version": m.version("pygod"),
    "files": {
        "pygod/nn/functional.py": {"path": F.__file__, "sha256": sha(F.__file__)},
        "pygod/nn/dominant.py": {"path": nn_dom.__file__, "sha256": sha(nn_dom.__file__)},
        "pygod/detector/dominant.py": {"path": det_dom.__file__, "sha256": sha(det_dom.__file__)},
        "pygod/detector/base.py": {"path": base.__file__, "sha256": sha(base.__file__)},
    },
}
out = sys.argv[1]
with open(out + ".json", "w", encoding="utf-8") as fh:
    json.dump(info, fh, indent=2)
with open(out + ".src.txt", "w", encoding="utf-8") as fh:
    fh.write("### DOMINANT.forward_model\n" + inspect.getsource(DOMINANT.forward_model))
    fh.write("\n### DOMINANTBase.__init__\n" + inspect.getsource(pygod.nn.DOMINANTBase.__init__))
    fh.write("\n### DOMINANTBase.forward\n" + inspect.getsource(pygod.nn.DOMINANTBase.forward))
    fh.write("\n### DeepDetector.fit\n" + inspect.getsource(base.DeepDetector.fit))
    fh.write("\n### DeepDetector.fit_inner\n" + inspect.getsource(base.DeepDetector.fit))
print(json.dumps(info, indent=2))

