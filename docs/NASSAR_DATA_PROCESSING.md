# Nassar et al. 2021 Data Processing Documentation

## Overview

This document details the data processing procedures used in Nassar et al. (2021) "All or nothing belief updating in patients with schizophrenia reduces precision and flexibility of beliefs", and how we replicate them for model validation.

## Data Structure

### Raw Data Files

**Location:** `C:/Users/aman0087/Documents/Github/Nassar_et_al_2021/Brain2021Code/realSubjects/`

**Structure:**
```
realSubjects/
├── Patients/           # Patient cohort 1
├── Patients2/          # Patient cohort 2
└── Normal Controls/    # Control subjects
```

### Subject Files

Each subject has **4 condition files**:
1. `{SubjectID}_cloud_cp_avoid_*.mat` - Changepoint, loss-avoidance framing
2. `{SubjectID}_cloud_cp_seek_*.mat` - Changepoint, gain-seeking framing
3. `{SubjectID}_cloud_drift_avoid_*.mat` - Oddball, loss-avoidance framing
4. `{SubjectID}_cloud_drift_seek_*.mat` - Oddball, gain-seeking framing

**Note:**
- `cp` = changepoint condition
- `drift` = oddball condition
- `avoid/seek` = task framing (not critical for analysis)
- `cloud` = helicopter hidden (vs `vis` = visible)

### File Contents

Each .mat file contains:
- **statusData**: Array of 100 blocks (1 trial per block)
- **payoutData**: Performance/reward information

**Key statusData fields:**
- `currentOutcome` - Bag landing position (outcome)
- `currentPrediction` - Bucket placement (agent's prediction)
- `currentUpdate` - Bucket movement (update)
- `currentDelta` - Prediction error (outcome - prediction)
- `currentMean` - True helicopter position
- `blockCompletedTrials` - Trial number within block
- `isChangeTrial` - Boolean flag for changepoint trials

---

## Data Cleaning Procedures

### From AASP_mastList.m

**Key parameters (lines 1-30):**
```matlab
drop=3;         % Drop first 3 trials from each block
perfThresh=32;  % Performance threshold
```

**Trial inclusion criteria (line 476):**
```matlab
isGood = all(isfinite(xes2), 2) & isfinite(PE) & isfinite(UP) & trialNum'>drop
```

**Translation:**
1. ✓ **Drop first N trials**: Remove first 3 trials from each 100-trial block
2. ✓ **Finite values only**: Exclude trials with NaN/Inf in prediction errors or updates
3. ✓ **Learning rate bounds**: Cap learning rates at [0, 1]
   ```matlab
   LR(LR>1)=1;
   LR(LR<0)=0;
   ```

### Data Concatenation

**Order matches AASP_mastList.m lines 221-270:**
```
[cloud_cp_avoid, cloud_cp_seek, cloud_drift_avoid, cloud_drift_seek]
```

This results in ~388 valid trials per subject (4 × 100 blocks - dropped trials).

---

## Fitted Parameters

### Parameter File

**Location:** `heliParamEstimatesForJim_23-Nov-2021.mat`

**Contents:**
- `paramLabels`: Parameter names
- `params`: Fitted values (134 subjects × 10 parameters)
- `subName`: Subject IDs
- `isPatient`: Boolean patient/control classification
- `isGood`: Data quality flag

### Parameter Mapping

| Nassar Label | Our Model | Description |
|--------------|-----------|-------------|
| HAZ | H | Hazard rate (changepoint probability) |
| LW | LW | Likelihood weight (surprise sensitivity) |
| UD | UU | Uncertainty discount/underestimation |
| UP STD | σ_motor | Update standard deviation (motor noise) |
| UP STDslope | σ_LR | Update variance slope (scales with LR) |
| pCE | - | Probability changepoint (derived) |
| binScale | - | Bin scaling (for sliding window analysis) |
| DS | - | Additional parameter |
| pCE CP | - | CP-specific changepoint prob |
| binSTD | - | Bin standard deviation |

**Note:** Nassar uses 10 parameters; our implementation uses 5 core parameters (H, LW, UU, σ_motor, σ_LR).

---

## Expected Data Characteristics

### Trial counts

- **Total subjects**: 134 (102 patients + 32 controls)
- **Conditions per subject**: 4
- **Trials per condition**: 100
- **Raw trials per subject**: 400
- **Valid trials per subject**: ~388 (after dropping first 3 per block)

### Data ranges

**Helicopter task:**
- Outcome space: [0, 300]
- Standard deviation of bag drop: σ_N = 20
- Hazard rate: H ≈ 0.125 (normative)
- High hazard comparison model: H = 0.4

### Quality checks

From our extraction:
- All trials should have finite prediction errors
- All trials should have finite updates
- Trial numbers should be > 3 (after drop period)
- Outcomes should be in [0, 300]
- Learning rates should be in [0, 1]

---

## Our Extraction Pipeline

### Script: `scripts/data_pipeline/extract_nassar_trials.py`

**Steps:**
1. Load all subject files (Patients, Patients2, Controls)
2. For each subject:
   - Load 4 condition files
   - Extract trial-by-trial data from 100-block arrays
   - Concatenate across conditions in correct order
   - Apply Nassar cleaning criteria
3. Save processed data:
   - `output/processed/nassar2021/subject_trials.npy`
   - `output/processed/nassar2021/subject_metadata.csv`

**Output format:**
```python
{
    'subject_id': 'SP_######',
    'is_patient': bool,
    'outcome': array,      # bag positions (for fitting)
    'prediction': array,   # bucket positions (for fitting)
    'update': array,       # bucket updates
    'delta': array,        # prediction errors
    'condition': array,    # 0=cp_avoid, 1=cp_seek, 2=drift_avoid, 3=drift_seek
    'n_trials': int,       # number of valid trials
}
```

---

## Validation Strategy

### Phase 1: ✓ Reproduce sliding window analysis
- Load pre-computed sliding window fits
- Reproduce Figure 6 (learning rate curves)
- Validate context discrimination metric
- **Status:** Complete (see `docs/BAYESIAN_VALIDATION.md`)

### Phase 2: Extract and fit trial-by-trial data (Current)
1. **Extract raw trials** ← We are here
   - Apply Nassar's data cleaning
   - Verify trial counts match expectations

2. **Fit our Bayesian model**
   - Use PyEM for fast fitting
   - Fit both CP and OB contexts separately
   - Extract fitted parameters

3. **Compare to Nassar's fitted parameters**
   - Load `heliParamEstimatesForJim_23-Nov-2021.mat`
   - Compare parameter values
   - Assess correlation and agreement

4. **Validate fitting procedure**
   - Check if our fitted params reproduce Nassar's
   - Verify model fit quality (NegLL, BIC)
   - Ensure parameter ranges are reasonable

### Phase 3: Apply to RNN agents
- Extract RNN behavioral data
- Fit using validated pipeline
- Compare RNN vs human parameters

---

## Key Differences: Nassar vs Our Implementation

### Model specification

| Aspect | Nassar et al. | Our Implementation |
|--------|---------------|-------------------|
| Parameters | 10 (includes bin scaling, pCE) | 5 (core normative model) |
| Contexts | CP and OB fit together | CP and OB fit separately |
| Priors | Custom priors from `modPriors/` | Weakly informative Beta/HalfNormal |
| Fitting | EM algorithm | PyEM (MLE) or NumPyro (MCMC) |

### Data processing

| Aspect | Nassar et al. | Our Implementation |
|--------|---------------|-------------------|
| Trial dropping | First 3 per block | Same |
| Concatenation | All 4 conditions | Same |
| LR bounds | [0, 1] | Same |
| Finite checks | Yes | Same |

---

## References

1. **Nassar, M. R., et al. (2021).** Brain, 144(3), 1013-1029.
2. **AASP_mastList.m** - Main analysis script
3. **loadApproachAvoidData.m** - Data loading function
4. **getTrialVarsFromPEs_cannon.m** - Model computation

---

## File Locations

### Input data
- Raw subject files: `C:/Users/aman0087/Documents/Github/Nassar_et_al_2021/Brain2021Code/realSubjects/`
- Fitted parameters: `C:/Users/aman0087/Documents/Github/Nassar_et_al_2021/Brain2021Code/heliParamEstimatesForJim_23-Nov-2021.mat`
- Sliding window fits: `data/raw/nassar2021/slidingWindowFits_*.mat`

### Our scripts
- Extraction: `scripts/data_pipeline/extract_nassar_trials.py`
- Validation: `scripts/analysis/validate_nassar2021.py`

### Output
- Processed trials: `output/processed/nassar2021/subject_trials.npy`
- Metadata: `output/processed/nassar2021/subject_metadata.csv`

---

**Last Updated:** 2025-01-23
**Status:** Phase 2 - Trial extraction in progress
