# A08 provider-derived feature provenance

Original provider transactions are available in `gog_round7_upstream/Token Data/transactions/{chain}.zip`, separate from historical converted graphs. The inspected CSV schema is block_number,from,to,transaction_hash,value,timestamp. Feature construction reads only from/to directly from every original CSV. The eight feature definitions, transforms, population and edge metric are in evidence/a08_data_repair/{feature_spec,relation_spec}.yaml. Constant-zero padding, Level1 score/logit and legacy embeddings are not read.

Input prediction unit is a token contract represented by its full static transfer history. Endpoint addresses are lowercased/trimmed and validated; the zero mint/burn address remains observed. Counts retain repeated and self transfers. Distinct directed pairs and reciprocity are derived by canonical factorization; no transaction value, timestamp or guessed legacy-column semantics is used. This is neither temporal inference nor a newly learned representation. All observed contract features fit the same label-blind transductive mean/std transform before model evaluation.

Provider README states Category0=fraud; original gog dataset/process_graph_metrics.py maps Category0 to binary1. A separate stable-ID join attaches these targets after observable construction. Missing labels are not benign0. Raw archive filenames provide the population, so availability differences relative to converted JSONs can change N/prevalence. New split membership and this correction are declared before observing predictive scores.

Raw ZIPs/member observations/addresses are author-local research-access material. No redistribution right is asserted; public numeric aggregate/hash evidence is separated from raw sources and manuscript files. Full member hashes and example row/count checks are available locally for audit.

## Independent real-source examples

Six actual provider CSVs (two per chain) were independently parsed with the standard CSV module and set arithmetic, without calling the feature builder. Every raw feature agrees within absolute1e-12. Full-population means/scales were independently recalculated using math.fsum; normalized float32 inputs agree within declared1e-6 absolute/relative tolerance. Example selection is lexicographic among nonempty members no larger than15KB, independent of labels and scores. Public audit contains hashes/error bounds; raw addresses and feature values remain local. See audit/raw_feature_examples.json. This source correctness check does not certify campaign completion.
