import time

def test_imp(name, stmt):
    t0 = time.time()
    exec(stmt)
    print(f"{name}: {time.time()-t0:.2f}s", flush=True)

test_imp("pygod.detector", "from pygod.detector import DeepDetector")
test_imp("pygod.nn.gadnr", "import pygod.nn.gadnr as m")
test_imp("pygod.utils", "from pygod.utils import logger")
test_imp("custom.gadnr", "import sys; sys.path.insert(0, 'src'); from gog_fraud.models.pygod.gadnr import GADNR")
print("All imports tested successfully!")
