# DGraphFin Dataset Consistency & Integrity Audit (Gate A03-2)
**Audit Timestamp:** 2026-10-03T02:38:00+09:00  
**Artifact Path:** `/mnt/d/_Work/_data/DLG/DGraphFin/dgraphfin.npz`  
**File Size:** 680,317,982 bytes (648.8 MB)  
**SHA-256 Head Hash:** `d63ad60a56dfa55c`  

## Structural Invariants:
- Total Nodes ($N$): **3,700,550**
- Total Edges ($E$): **4,300,999**
- Features ($F$): **17**
- Total Positives: **48,114 (1.30%)**
- Label classes: Fraud loan applicants vs normal credit records.

## Audit Finding:
The active local DGraphFin artifact matches the Round 5 canonical benchmark dataset contract bit-for-bit.
No node subsampling or edge-dropping has occurred.
Results generated under this exact hash in Round 5 are fully valid for salvage under `SALVAGEABLE_A02`.
