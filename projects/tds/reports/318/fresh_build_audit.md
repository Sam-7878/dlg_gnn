# Fresh-build audit

Each supplied post-declaration and new current venue ZIP was extracted into its own empty temporary directory.
Article and Supplement were independently compiled: eight successful cold builds total.
SOURCE_DATE_EPOCH pins 2026-10-30. Byte-hash differences are explicitly retained; extracted text must match.

|Venue|Generation|Article / Supplement pages|Article bytes equal|Supplement bytes equal|Text equal|Overfull|
|---|---|---|---|---|---|---|
|preprints|post_declaration|13 / 4|False|False|True|0|
|preprints|current|13 / 4|True|True|True|0|
|applsci|post_declaration|13 / 4|False|False|True|0|
|applsci|current|13 / 4|True|True|True|0|

Undefined citations/references and missing-asset compilation failures: zero.
Citation key order and figure/table labels match across wrappers; shared Bib/figures/tables/declarations are byte-identical.
Build metadata and full per-artifact hashes: results/318/build_verification.json.
