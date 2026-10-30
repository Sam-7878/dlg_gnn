#!/usr/bin/env python3
"""
Gate A22-G8: Final Package & Hash Synchronization
Re-packages source_bundle.zip and generates SHA256SUMS.txt for preprints and applsci.
"""

import os
import hashlib
import zipfile

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def package_preprints():
    base_dir = "/mnt/d/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/preprints"
    zip_path = os.path.join(base_dir, "source_bundle.zip")
    
    files_to_zip = [
        "DLG-Benchmark_A16_Preprints.tex",
        "references.bib",
        "figures/fig_conditional_delta.pdf",
        "figures/fig_conditional_delta.png",
        "figures/fig_dlg_architecture.pdf",
        "figures/fig_dlg_architecture.png",
        "figures/fig_support_matrix.pdf",
        "figures/fig_support_matrix.png"
    ]
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path in files_to_zip:
            full_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Missing file for zip: {full_path}")
            zf.write(full_path, arcname=rel_path)
            
    print(f"Packaged {zip_path} ({os.path.getsize(zip_path):,} bytes)")
    
    # Generate SHA256SUMS.txt
    artifacts = [
        "DLG-Benchmark_A16_Preprints.pdf",
        "DLG-Benchmark_A16_Preprints.tex",
        "references.bib",
        "source_bundle.zip"
    ]
    sums_lines = []
    for art in artifacts:
        p = os.path.join(base_dir, art)
        sha = compute_sha256(p)
        sums_lines.append(f"{sha}  {art}\n")
        
    sums_path = os.path.join(base_dir, "SHA256SUMS.txt")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.writelines(sums_lines)
    print(f"Wrote {sums_path}:\n" + "".join(sums_lines))

def package_applsci():
    base_dir = "/mnt/d/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/applsci"
    zip_path = os.path.join(base_dir, "source_bundle.zip")
    
    files_to_zip = [
        "DLG-Benchmark_A16_AppliedSciences.tex",
        "references.bib",
        "soul.sty",
        "soul-ori.sty",
        "Definitions/journalnames.tex",
        "Definitions/logo-mdpi-eps-converted-to.pdf",
        "Definitions/logo-mdpi.eps",
        "Definitions/logo-orcid.pdf",
        "Definitions/logo-updates-eps-converted-to.pdf",
        "Definitions/logo-updates.eps",
        "Definitions/mdpi.bst",
        "Definitions/mdpi.cls",
        "Definitions/mdpi_apacite.bst",
        "Definitions/mdpi_apacite.sty",
        "Definitions/mdpi_chicago.bst",
        "figures/fig_conditional_delta.pdf",
        "figures/fig_conditional_delta.png",
        "figures/fig_dlg_architecture.pdf",
        "figures/fig_dlg_architecture.png",
        "figures/fig_support_matrix.pdf",
        "figures/fig_support_matrix.png"
    ]
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_path in files_to_zip:
            full_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Missing file for zip: {full_path}")
            zf.write(full_path, arcname=rel_path)
            
    print(f"Packaged {zip_path} ({os.path.getsize(zip_path):,} bytes)")
    
    # Generate SHA256SUMS.txt
    artifacts = [
        "DLG-Benchmark_A16_AppliedSciences.pdf",
        "DLG-Benchmark_A16_AppliedSciences.tex",
        "references.bib",
        "source_bundle.zip"
    ]
    sums_lines = []
    for art in artifacts:
        p = os.path.join(base_dir, art)
        sha = compute_sha256(p)
        sums_lines.append(f"{sha}  {art}\n")
        
    sums_path = os.path.join(base_dir, "SHA256SUMS.txt")
    with open(sums_path, "w", encoding="utf-8") as f:
        f.writelines(sums_lines)
    print(f"Wrote {sums_path}:\n" + "".join(sums_lines))

if __name__ == "__main__":
    package_preprints()
    package_applsci()

