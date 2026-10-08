# P2 Audit Report 08: MDPI Special Issue Post-Preprint Submission Readiness

## 1. Executive Summary
- **Target Journal**: *MDPI Applied Sciences* (ISSN 2076-3417).
- **Special Issue**: *"Graph Neural Networks: Theory, Methods and Applications"*
  - Special Issue URL: `https://www.mdpi.com/journal/applsci/special_issues/C80IXAF9V4`
  - Deadline: 20 November 2026
- **Status**: **STAGED FOR POST-PREPRINT SUBMISSION**.

---

## 2. Transition Procedure (Preprint Deposited -> MDPI Submission)
Once the preprint is posted and Preprints.org issues the new benchmark DOI:
1. Run the synchronization tool:
   ```bash
   python scripts/publication/update_preprint_doi.py --doi 10.20944/preprints2026XX.XXXX.v1
   ```
2. Re-run `scripts/manuscript/build_mdpi_bundle_p1.py` to bake the DOI into the final submission bundle.
3. Submit `publication/benchmark/mdpi/DLG_Benchmark_MDPI_Submission.zip` and `publication/benchmark/mdpi/DLG-Benchmark.pdf` to *Applied Sciences*.
4. On the MDPI submission portal:
   - Select Special Issue: *"Graph Neural Networks: Theory, Methods and Applications"*.
   - Declare the Preprints.org deposit under CC BY 4.0 license.
   - Upload `publication/benchmark/mdpi/cover_letter.md`.

---

## 3. Final State Declaration
Following preprint deposit and DOI synchronization, the state will transition to:

$$\mathbf{PREPRINT\_POSTED\_READY\_FOR\_MDPI\_SUBMISSION}$$
