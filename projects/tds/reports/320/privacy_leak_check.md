# Privacy & Secret Leak Check — R320

Audit performed on the R320 submission and release bundles.

## Check Items & Results
1. **API Keys & Credentials**: 0 secrets detected. No OpenAI, Anthropic, or cloud provider API tokens present in submission sources.
2. **Local File Paths**: No hardcoded local machine user paths (`C:\Users\...` or `/mnt/d/...`) present in public manuscript source files. Relative paths (`figures/...`, `data/...`) used throughout LaTeX documents.
3. **Wallet Addresses & PII**: Raw wallet hexadecimal addresses and transaction hashes are excluded from the manuscript and the public release payload. Sample IDs are numeric indices.
4. **Author Email & Affiliation**: Verified against institutional domains (`ajou.ac.kr`). Both authors correctly assigned to `Department of Computer Engineering, Graduate School, Ajou University`.
5. **Verdict**: **PASS (NO LEAKS DETECTED)**.
