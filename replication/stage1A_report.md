# Stage 1A Audit Report

## Environment

- Python version: 3.11.15
- nibabel version: NOT INSTALLED
- numpy version: NOT INSTALLED
- pillow version: NOT INSTALLED
- Operating system: Linux-6.18.5-x86_64-with-glibc2.39
- Execution date and time: 2026-07-10 06:57:57
- OASIS root path: /home/user/chamak-repli/data/OASIS-1
- Clinical file path: /home/user/chamak-repli/data/oasis_cross-sectional.csv
- Random seed that will be used: 42

```
=== STAGE 1A: DATASET AUDIT ===
Date: 2026-07-10 06:57:57
OASIS Root: /home/user/chamak-repli/data/OASIS-1
Clinical File: /home/user/chamak-repli/data/oasis_cross-sectional.csv

--- Disc Directories Found ---
Total disc directories: 0

--- Subject Directories ---
Total subject directories found: 0

--- Volume File Audit ---
Subjects with exactly 1 volume file:  0
Subjects with more than 1 volume file: 0
Subjects with 0 volume files:          0

--- Clinical File ---
File found: NO
Filename: oasis_cross-sectional.csv
Size: N/A
Readable: NO

--- COMPARISON TO PAPER ---
| Metric                          | Paper (§3.1) | Observed | Match |
|---------------------------------|--------------|----------|-------|
| Total subjects with directories | 235          | 0        | NO    |
| Subjects with volume file found | 235          | 0        | NO    |

--- STOP CONDITIONS ---
STOP
STOP: Subject count mismatch
Exact count found: 0
  - OASIS root does not exist or is not a directory: /home/user/chamak-repli/data/OASIS-1
  - Subject directory count is 0, expected 235.
  - Subjects with a volume file is 0, expected 235.
  - Clinical file not found at: /home/user/chamak-repli/data/oasis_cross-sectional.csv
```
