# Round P3 Audit Report 04: Cover Letter State & Claim Audit

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite  
**Scope:** Cover letter claim boundaries, dataset naming provenance, preprint state machine management, and author signature blocks.

---

## 1. Softening of Dataset-Dependence Claims

In Round P2, the cover letter retained an overly definitive phrasing:
> *"paired sensitivity analysis demonstrates that the utility of local neighborhood augmentation is strictly dataset-dependent"*

Following Round P3 editorial guidelines, the language was moderated to avoid unhedged causal and absolute claims:
> *"the primary benchmark and targeted sensitivity analyses show that the utility of local neighborhood augmentation is dataset dependent (improving PR-AUC on Elliptic by $+0.0350$, but reducing it on dense graphs such as Reddit-Syn by $-0.0656$)"*

- Absolute adverbs (`strictly`) and deductive claims (`demonstrates`) were removed.
- **Automated Test:** `tests/benchmark/publication_p3/test_cover_letter_bounded_dataset_dependence_claim.py` (PASS)

---

## 2. Dataset Display & Provenance Reconciliation

The dataset enumeration in Contribution 1 of the cover letter was aligned with publication-wide provenance standards:
- Declares **10 primary graphs** (3 real-label + 7 synthetic-injection).
- Names all three real financial graphs: `Elliptic, DGraphFin, and BitcoinOTC`.
- Names all seven synthetic graphs with `-Syn`: `Yelp-Syn, Amazon-Syn, Flickr-Syn, Reddit-Syn, Cora-Syn, CiteSeer-Syn, and PubMed-Syn`.
- Explicitly states that synthetic graphs are constructed from public base graphs `Yelp, Amazon, Flickr, Reddit, Cora, CiteSeer, and PubMed`.
- **Automated Test:** `tests/benchmark/publication_p2/test_cover_letter_dataset_count.py` (PASS)

---

## 3. Preprint State Machine Management

To prevent submitting a cover letter to MDPI that prematurely asserts a completed deposit or fabricates a DOI before deposit screening:
- **Preprint Status:** Declared as *"Preprints.org deposit initiated; DOI will be declared upon official deposit confirmation."*
- **Companion Repository URL:** Declared as `https://github.com/Sam-7878/dlg_gnn` with tag `v1.0.0-preprint`.
- **Legacy / Preceding DOI Separation:** Verified that the DOI of the preceding architecture paper (`10.20944/preprints202609.0848.v1`) is not mistakenly assigned as the DOI of this follow-up benchmark paper.
- **Post-Deposit Flow:** The MDPI bundle is configured for immediate post-deposit recompilation once the official Preprints.org DOI is assigned.
- **Automated Test:** `tests/benchmark/publication_p3/test_cover_letter_preprint_state_machine.py` (PASS)

---

## 4. Author Signature Block & ORCIDs

The signature block of `publication/benchmark/mdpi/cover_letter.md` has been verified with authentic institutional affiliations and ORCIDs:
- **Corresponding Author:** Prof. Ki-Hyung Kim (`kkim86@ajou.ac.kr`, Tel: +82-31-219-2433, ORCID: `0000-0002-2321-4475`)
- **First Author:** SeongSu Park (`parky@ajou.ac.kr`, ORCID: `0009-0008-4056-3875`)

---

## 5. Audit Verdict
**STATUS: PASS (`READY_FOR_PREPRINTS_ORG_DEPOSIT`)**  
The cover letter is scientifically bounded, state-machine clean, and ready for staging.
