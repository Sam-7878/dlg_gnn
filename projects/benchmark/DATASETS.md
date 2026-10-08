# Benchmark data acquisition and construction

The paths and hashes below come from the frozen A05 canonical manifest in `evidence/frozen_a05_a06_evidence.zip`. Upstream license and access terms must be checked at the provider before downloading; third-party raw datasets are not redistributed in this repository. A raw hash identifies the exact local input used by this release and may differ from a newly downloaded provider package.

**Access limitation:** The Ethereum, BSC and Polygon entries are locally constructed `*_hybrid_graph.pt` artifacts. The frozen manifest identifies their hashes and loader, but does not establish a public upstream download or redistribution permission. The repository URL shown below is a provenance/code location, not a dataset download. Consequently, an independent full rerun of those three graphs is not yet supported from public raw data alone; the evidence-only `verify` and `paper` paths remain available. LANL's listed `.pt` is also a constructed graph; obtain source logs from LANL subject to its terms.

Graph construction uses `evaluation/benchmark/v2/scripts/a04_build_dataset_manifest.py` and the frozen source snapshot/configs in the evidence ZIP. Follow `projects/benchmark/README.md` for the evidence-only reviewer path. The original raw path is retained for provenance, not as a portable path.

## Elliptic

- Upstream landing/provider: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `elliptic_txs_classes.csv`
- Frozen raw SHA-256: `93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/Elliptic.pt`
- Constructed file SHA-256: `945b96f137483a1cdb9c1728798429e2d6a45696dcd07146b5cdf42734b67826`
- Constructed tensor SHA-256: `c1f46f6112f360dc8a23768719cea9669eb03d1317a366e4ad80b82f9f519690`
- Evaluation nodes: 46564; positives: 4545; unit: labeled_transaction_node
- Label provenance: `real_external_label`
- Source evidence: benchmark_8x10_pipeline.load_elliptic (unknown labels removed)

## DGraphFin

- Upstream landing/provider: https://dgraph.xinye.com/dataset
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `dgraphfin.npz`
- Frozen raw SHA-256: `95470dab2c48523f7118a92204c090de37a957bb053bd5841c7bdba09558ba85`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/DGraphFin.pt`
- Constructed file SHA-256: `074c51214dbbc6028ad35b40b5447330bbaccce998f976156533fce4b0c2066e`
- Constructed tensor SHA-256: `6fb60ea28d75eff20016a1064d669f7cc12720216bf5283a2d7864938590f923`
- Evaluation nodes: 367702; positives: 4652; unit: loan_account_node
- Label provenance: `real_external_label`
- Source evidence: dgraphfin_aligned.load_dgraphfin_aligned; official val/test masks
## BitcoinOTC

- Upstream landing/provider: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `data.pt`
- Frozen raw SHA-256: `e39535e2040d94d5f522ca97cbb0cb2aef0aa669f2b2028b2975913e3397a074`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/BitcoinOTC.pt`
- Constructed file SHA-256: `47b6ccbf1c570e54aafa29af8dd1ae411196ecfe0eb858f8d0e3f6b771f19052`
- Constructed tensor SHA-256: `35d3fa39fff6e8b97194ed6cde791585368dc7bf862eca219898baf84030af32`
- Evaluation nodes: 6005; positives: 180; unit: node
- Label provenance: `synthetic_node_injection_on_real_trust_graph`
- Source evidence: benchmark_8x10_pipeline.load_bitcoinotc + fixed injection seed 42; differs from archived round5 data_freeze.json; A05 superseding 35-run canonical-tensor campaign

## Ethereum

- Upstream landing/provider: https://github.com/Sam-7878/dlg_gnn
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `ethereum_hybrid_graph.pt`
- Frozen raw SHA-256: `4fa51d3e5fd09464f6ef7bb106376b365758e1e35d580c46c6cea65b4261c654`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/Ethereum.pt`
- Constructed file SHA-256: `f313b2ac539a3da3645b262fd430cc909e857666af120fdb0bc2d1fff1e643ae`
- Constructed tensor SHA-256: `f4b6dd781059ad9f89be4a6810dcd8ab383eab02a8aca5ed7b5b2075c612f39d`
- Evaluation nodes: 14385; positives: 6018; unit: smart_contract_node
- Label provenance: `real_external_label`
- Source evidence: a03_run_crypto_production.load_gog_graph + seed split

## BSC

- Upstream landing/provider: https://github.com/Sam-7878/dlg_gnn
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `bsc_hybrid_graph.pt`
- Frozen raw SHA-256: `2bd0a4943829241b84c694c1b301634e876e310bd63d415173a36de3ee510883`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/BSC.pt`
- Constructed file SHA-256: `ee8623293f23cbf56eec6119acccf57e594ba86d2e0568af103e780766b7f774`
- Constructed tensor SHA-256: `01f67876c85ec972d4fd4208a9e9e1af12289d2778864529be15d164be4709d5`
- Evaluation nodes: 7481; positives: 1104; unit: smart_contract_node
- Label provenance: `real_external_label`
- Source evidence: a03_run_crypto_production.load_gog_graph + seed split

## Polygon

- Upstream landing/provider: https://github.com/Sam-7878/dlg_gnn
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `polygon_hybrid_graph.pt`
- Frozen raw SHA-256: `292528febddc636ee2b5b09a76f7250e4473ff6d518d7977c49ff0beae7dad74`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/Polygon.pt`
- Constructed file SHA-256: `2da5bcb770b60bdf0ca771e4806b9eb9eeb087cc62eb1b622456b2a463ac78ae`
- Constructed tensor SHA-256: `f4649616c7855e2ef4783db848e72b10ca8fccb48f329d9df566fbece2702303`
- Evaluation nodes: 2303; positives: 60; unit: smart_contract_node
- Label provenance: `real_external_label`
- Source evidence: a03_run_crypto_production.load_gog_graph + seed split

## Yelp-Syn

- Upstream landing/provider: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Yelp.html
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `data.pt`
- Frozen raw SHA-256: `6ef8da6d449810b2a0f492fa7da30c0cac3cdf4cfb8051286acbac2f7a996b24`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/Yelp-Syn.pt`
- Constructed file SHA-256: `ace899104bb119c00a19d948812606920e78c2baa71b6b2b004871476f575a53`
- Constructed tensor SHA-256: `0dda5f7df4e8f01f17f38d32275801da92ccdd441f98dfa5f2e0b7eb9ad2c610`
- Evaluation nodes: 716847; positives: 7168; unit: node
- Label provenance: `synthetic_node_injection`
- Source evidence: benchmark_8x10_pipeline.load_yelp + fixed injection seed 42; all 3 tensor hashes match round5 data_freeze.json
## Amazon-Syn

- Upstream landing/provider: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Amazon.html
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `data.pt`
- Frozen raw SHA-256: `b54bd5fef41bd801c8bdce6140e9cea673c963f73459ee69d196079e6a6bf445`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/Amazon-Syn.pt`
- Constructed file SHA-256: `2cc7e9df096a0b9f41ae8f7a66bd27f3aae2971185ecfebbda140f6b6a4f9bbd`
- Constructed tensor SHA-256: `86219cbe56272a82f22de3b89d55563ece9d26816f0642b7f9dccc2d16cc04a8`
- Evaluation nodes: 13752; positives: 412; unit: node
- Label provenance: `synthetic_node_injection`
- Source evidence: benchmark_8x10_pipeline.load_amazon + fixed injection seed 42; all 3 tensor hashes match round5 data_freeze.json
## Reddit-Syn

- Upstream landing/provider: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Reddit.html
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `data.pt`
- Frozen raw SHA-256: `1de4167e65c8c675a4739924ec44736175fa13efc4f75e0f05c5cb79a4522582`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/Reddit-Syn.pt`
- Constructed file SHA-256: `15aa429034d64a452aa8a1ccad8b9fa1f8040fbe4bf0cbdf267bff078c6ba3fa`
- Constructed tensor SHA-256: `3cef16a021e97c6424ac70ef801ec7f0a6f96a12d526463a0e500b00fa43a90f`
- Evaluation nodes: 232965; positives: 4659; unit: node
- Label provenance: `synthetic_node_injection`
- Source evidence: benchmark_8x10_pipeline.load_reddit + fixed injection seed 42; all 3 tensor hashes match round5 data_freeze.json
## Flickr-Syn

- Upstream landing/provider: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Flickr.html
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `data.pt`
- Frozen raw SHA-256: `223c1cd0c18d80cda66651192346754504c708764be5c8dc99d9addd38cb9d69`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/Flickr-Syn.pt`
- Constructed file SHA-256: `62d9efcaa3f36da29147720a14e5efe0b0ee8f425b96ff3a2a0b48e5d0f7ac75`
- Constructed tensor SHA-256: `e4be871167ae2e7a067221c22c8fac902bf2290bfbb11d6a31e34c7c4fee1893`
- Evaluation nodes: 89250; positives: 1785; unit: node
- Label provenance: `synthetic_node_injection`
- Source evidence: benchmark_8x10_pipeline.load_flickr + fixed injection seed 42; all 3 tensor hashes match round5 data_freeze.json
## Cora-Syn

- Upstream landing/provider: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Planetoid.html
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `data.pt`
- Frozen raw SHA-256: `1ac38ca581468b1c24a6a14ca30f735f384ff8c643b781c657c3d476e382413f`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/Cora-Syn.pt`
- Constructed file SHA-256: `064175ba2f9f0530e0e8b9809e69d6e49e75fcf50944ae627688d4f8c1e81e44`
- Constructed tensor SHA-256: `b203ae770bd207070c578b3bc4fe69b073591bb119fe78290d3bbd5489f85548`
- Evaluation nodes: 2708; positives: 81; unit: node
- Label provenance: `synthetic_node_injection`
- Source evidence: benchmark_8x10_pipeline.load_cora + fixed injection seed 42; all 3 tensor hashes match round5 data_freeze.json
## CiteSeer-Syn

- Upstream landing/provider: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Planetoid.html
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `data.pt`
- Frozen raw SHA-256: `d40ff756d7134c211899c75d52b1ad3c2429bb1880d84eb0a129317df448534d`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/CiteSeer-Syn.pt`
- Constructed file SHA-256: `392d7b0add1af6d37f49ea76ffc584f8caf7d50ea1b396dd4f04dcc43bba1d49`
- Constructed tensor SHA-256: `f751616b6169181cef291cabadc9d0b6264d1e918a9cd4528b60696e235027bb`
- Evaluation nodes: 3327; positives: 99; unit: node
- Label provenance: `synthetic_node_injection`
- Source evidence: benchmark_8x10_pipeline.load_citeseer + fixed injection seed 42; all 3 tensor hashes match round5 data_freeze.json
## PubMed-Syn

- Upstream landing/provider: https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Planetoid.html
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `data.pt`
- Frozen raw SHA-256: `2b28d1f66a49c05ab1422bc5653fb7f7e1b3243c644b3b8fb12fadc562b3d09b`
- Constructed artifact: `outputs/benchmark/a04_constructed_graphs/PubMed-Syn.pt`
- Constructed file SHA-256: `7c781388caa4d7b6b03fbe54ec975a01081597c7b24200174c5c9cfcca12a354`
- Constructed tensor SHA-256: `33dbbb6627da485998ea89b54bb5ab522e32f73b8e6db64b42dcac5e51966e6b`
- Evaluation nodes: 19717; positives: 591; unit: node
- Label provenance: `synthetic_node_injection`
- Source evidence: benchmark_8x10_pipeline.load_pubmed + fixed injection seed 42; all 3 tensor hashes match round5 data_freeze.json
## LANL-RedTeam

- Upstream landing/provider: https://csr.lanl.gov/data/cyber1/
- Access and license: check provider terms; manual acquisition unless the provider package loader is available.
- Expected raw basename: `lanl_graph.pt`
- Frozen raw SHA-256: `689c2968fe3ece9494196515e6089d6db3f430530e55b8b410d116b27c920359`
- Constructed artifact: `outputs/benchmark/sci_defense_extension_real/graphs/lanl_graph.pt`
- Constructed file SHA-256: `689c2968fe3ece9494196515e6089d6db3f430530e55b8b410d116b27c920359`
- Constructed tensor SHA-256: `2a1531a4543ecc3892f3a2d493838ded2985ede80b18cba8fa5ee0dcbf8ae392`
- Evaluation nodes: 16694; positives: 301; unit: destination_computer_node
- Label provenance: `real_external_label`
- Source evidence: lanl_graph.pt + lanl_ground_truth_freeze.json; 749 red-team events map to 301 nodes
