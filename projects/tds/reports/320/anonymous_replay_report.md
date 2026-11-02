# Anonymous Replay Report — R320 (Pending Release)

- **Target Repository**: `https://github.com/Sam-7878/dlg_gnn`
- **Current Status**: **PENDING_PUBLIC_DISPATCH**
- **Verification Plan**:
  1. Once authorized, tag `tds-r320-v1.0.0` will be pushed.
  2. A clean non-authenticated environment (isolated container or ephemeral VM) will clone the tag.
  3. `python replay_scores.py` will execute without credentials to reproduce all primary tables and figures.
  4. L1 numerical replay reproducibility will be confirmed.
