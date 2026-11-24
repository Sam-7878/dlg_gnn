# Note on the `memory_cap` field (2026-11-24)

`a29_dense_vs_gram.py` and `a29_scaling_sweep.py` call `torch.cuda.set_per_process_memory_fraction(1.0, 0)` to stop the
Windows/WSL driver from letting allocations spill past the 24 GB RTX 3090 into shared system memory. The call had no effect
under WSL: on Elliptic the dense DOMINANT run had 32.52 GiB allocated before failing on a further 8.08 GiB request, and the
dense DLG-Base run completed with a 32.54 GiB allocated / 32.74 GiB reserved peak. The `memory_cap` string in the JSON
records therefore states the intent, not the effect. All analyses treat a run as spilled when its peak reserved memory
exceeds the physical 24 GiB (`torch.cuda.get_device_properties(0).total_memory`), and the scaling sweep stops a model at its
first spilled size.
