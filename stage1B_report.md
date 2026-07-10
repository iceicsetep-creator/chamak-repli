=== STAGE 1B: CLINICAL LABELS ===
Executed: 2026-07-10 07:23:12

--- Environment ---
Python: 3.11.15
Pandas: 3.0.3

--- Input ---
File: oasis_cross-sectional-5708aa0a98d82080.xlsx
Rows loaded: 436
Columns: ['ID', 'M/F', 'Hand', 'Age', 'Educ', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF', 'Delay']

--- CDR Exclusion ---
Rows removed (CDR missing): 201
Rows remaining: 235

--- Binary Label Assignment ---
Rule: CDR == 0.0 → No_Dementia; CDR > 0.0 → Dementia
(Research Supervisor decision resolving paper §3.1 ambiguity)

--- CDR Distribution ---
CDR = 0.0 (No_Dementia):  135 subjects
CDR = 0.5 (Very Mild):    70 subjects
CDR = 1.0 (Mild):         28 subjects
CDR = 2.0 (Moderate):     2 subjects

--- PAPER COMPARISON TABLE ---
| Metric                  | Paper (Fig.1 / §3.1) | Observed | Match  |
|-------------------------|----------------------|----------|--------|
| Total after CDR filter  | 235                  | 235      | YES    |
| No_Dementia (CDR=0.0)   | 135                  | 135      | YES    |
| Dementia (CDR>0)        | 100                  | 100      | YES    |
| Very Mild (CDR=0.5)     | 70                   | 70       | YES    |
| Mild (CDR=1.0)          | 28                   | 28       | YES    |
| Moderate (CDR=2.0)      | 2                    | 2        | YES    |

--- Verification ---
Total count check (235):     PASSED
No_Dementia count (135):     PASSED
Dementia count (100):        PASSED
Duplicate subject ID check:  PASSED

--- Output ---
Saved: /kaggle/working/subject_labels.csv (235 rows)

--- RESULT ---
[PROCEED]
