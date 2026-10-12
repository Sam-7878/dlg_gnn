# R02 reference and citation audit (12 October 2026)

The main manuscript cites 20 entries, expanded from the original six; original
method citations are GIN (Xu et al., ICLR 2019), GATv2 (Brody et al., ICLR 2022),
Platt's 1999 manuscript, MC dropout (Gal/Ghahramani, ICML 2016) and calibration
(Guo et al., ICML 2017). Final actual BibTeX/PDF counts and visible DOI/URLs are
bound to `paper_build_audit.json` and the separately inspected final PDF hashes.
Numeric natbib/unsrtnat printing makes URL/DOI fields visible, not merely stored
in uncited BibTeX. No external abstracts are copied into the paper.

Primary metadata sources checked for added references:

| Key | Primary source | Placement / scope |
|---|---|---|
| xu2019gin | https://arxiv.org/abs/1810.00826v3 (PDF states ICLR 2019) | Local architecture; not a new theoretical result here |
| brody2022gatv2 | https://arxiv.org/abs/2105.14491v3 | Independent relational stage / optional local control |
| platt1999probabilistic | https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf | Original probabilistic-output manuscript; title/author/1999 identity retained, no unverified DOI/venue/page range |
| gal2016dropout | https://proceedings.mlr.press/v48/gal16.html | Dropout stochastic passes, not universal uncertainty/risk validity |
| guo2017calibration | https://proceedings.mlr.press/v70/guo17a.html | Calibration context, not distribution-free early-exit guarantee |
| hamilton2017graphsage | https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html | Scalable inductive graph background |
| zeng2020graphsaint | https://arxiv.org/abs/1907.04931v4 | Sampling context, not an experimentally evaluated baseline |
| liu2022bond | https://proceedings.neurips.cc/paper_files/paper/2022/hash/acc1ec4a9c780006c9aafd595104816b-Abstract-Datasets_and_Benchmarks.html | Static unsupervised anomaly interface distinction |
| tang2023gadbench | https://proceedings.neurips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html | Supervised anomaly evaluation context |
| liu2024pygod | https://www.jmlr.org/papers/v25/23-0963.html | JMLR 25(141):1–9; official author Ruitong Zhang, not blindly copied Benchmark metadata |
| xu2020tgat | https://arxiv.org/abs/2002.07962 | Temporal neighborhood model versus checkpoint replay |
| rossi2020tgn | https://arxiv.org/abs/2006.10637v3 | Dynamic temporal memory; labeled workshop manuscript, not invented conference venue |
| weber2019elliptic | https://arxiv.org/abs/1908.02591 | Blockchain graph application; transaction-level targets not this paper's contract labels |
| errica2020fair | https://arxiv.org/abs/1912.09893 | Reproducible/task-matched comparison discipline |

Benchmark reference themes reused where relevant: scalable/inductive graphs,
BOND/GADBench/PyGOD, temporal graphs and fair evaluation. No Benchmark metric,
split, feature or protocol becomes StreamMC evidence by citation. Existing
GoG, three early-exit studies, DLG-GNN v1 and author-local Benchmark references
are retained with their existing distinct status. DLG-GNN DOI is exactly
`10.20944/preprints202609.0848.v1`, explicitly not peer-reviewed; Benchmark has
no invented DOI/publication status. No universal MC superiority is inferred.

Access limitation: the Colorado-hosted Platt PDF timed out. A primary-author
ResearchGate PDF was accessible to the browsing tool but text extraction was
garbled; title/date were index-visible. This is not recorded as a full-text
visual/source-access PASS. Unverified book pagination/DOI are deliberately not
asserted. The cited historical original-method identity is separate from a
promise that this particular mirror will remain downloadable.

Publisher pages for Applied Sciences were rate/access limited (429), so their
deadline/total submission-byte rule is an author pre-submission recheck, not
silently replaced by another MDPI journal's instructions.
