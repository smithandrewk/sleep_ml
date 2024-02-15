# BDHSC Case Competition

## Downloading Data
```
wget bdhsc_2024.tar.gz
gunzip bdhsc_2024.tar.gz
tar -xvf bdhsc_2024.tar
```

# Data Structure
## Data Directory Structure
```
bdhsc_2024/
├── README.md
├── stage1_labeled
│   ├── 0_0.csv
│   ├── 0_1.csv
│   ├── 1_0.csv
│   ├── 1_1.csv
│   ├── 2_0.csv
│   ├── 2_1.csv
│   ├── 3_0.csv
│   ├── 3_1.csv
│   ├── 4_0.csv
│   ├── 4_1.csv
│   ├── 5_0.csv
│   ├── 5_1.csv
│   ├── 6_0.csv
│   ├── 6_1.csv
│   ├── 7_0.csv
│   └── 7_1.csv
└── stage2_unlabeled
    ├── 10_0.csv
    ├── 10_1.csv
    ├── 11_0.csv
    ├── 11_1.csv
    ├── 12_0.csv
    ├── 12_1.csv
    ├── 13_0.csv
    ├── 13_1.csv
    ├── 14_0.csv
    ├── 14_1.csv
    ├── 15_0.csv
    ├── 15_1.csv
    ├── 8_0.csv
    ├── 8_1.csv
    ├── 9_0.csv
    └── 9_1.csv

2 directories, 33 files
```
## Data Naming
Data are named using the following format:
```
{animal_id}_{recording_id}.csv
```

**animal_id** $\in \{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\}$ and

**recording_id** $\in \{0,1\}$.

This means there are 16 animals who have each been recorded 2 times. 8 animal's recordings are provided in **stage1_labeled** as labeled data. 8 animal's recordings are provided in **stage2_unlabeled** as labeled data. 

## Stage 1: Labeled EEG Data
```
/bdhsc_2024/stage1_labeled
```
- Files: 16
- Animal Subjects: 8
- Each subject has been studied under 2 distinct experimental conditions, leading to the creation of 2 files per animal.
- Every animal subject has been evaluated under these conditions, contributing to the diversity of the dataset.
- Data are provided in csv format
- Each recording has 8640 rows and 5001 columns
- The first 5000 columns comprise the **features**
- The last column comprise the **labels**

### Data Content:
EEG signals that have been meticulously recorded and segmented into 10-second epochs.
Each epoch is annotated with a sleep-stage label, providing valuable insights into the behavioral states of the subjects.

### Labels:
- 0: Paradoxical sleep, often associated with REM sleep where dreaming occurs.
- 1: Slow-wave sleep, indicative of deep, restorative sleep stages.
- 2: Wakefulness, representing periods of active, conscious states.

## Stage 2: Unlabeled EEG Data
As with Stage 1, each of these new subjects has been studied under the same 2 experimental conditions, but the data in this stage are unlabeled.
```
/bdhsc_2024/stage2_unlabeled
```
- Files: 16
- Animal Subjects: 8
- Each subject has been studied under 2 distinct experimental conditions, leading to the creation of 2 files per animal.
- Every animal subject has been evaluated under these conditions, contributing to the diversity of the dataset.
- Data are provided in csv format
- Each recording has 8640 rows and 5000 columns
- All 5000 columns comprise the **features**

### Data Content:
EEG signals segmented into 10-second epochs, akin to Stage 1.
Unlike Stage 1, these epochs are not labeled, posing a challenge for participants to infer sleep stages or other underlying patterns.

### Data Submission
You must provide your submissions in a single directory called **stage1** that you must tar gzip to submit. An example scoring file is provided at /bdhsc_2024/example_scoring.csv. Each file should follow the following format:
- csv format
- a single column with 0, 1, or 2 for each sample
- no header
- naming scheme {animal_id}_{recording_id}.csv

# Data Delivery
The data will be delivered to the competition participants in two stages, aligning with the structure outlined above. This phased approach allows participants to initially work with labeled data to develop and train models, and subsequently apply or validate these models using the unlabeled data in Stage 2. This format is designed to test the participants' abilities in both supervised learning with Stage 1 and unsupervised or semi-supervised learning techniques in Stage 2, fostering a comprehensive analytical and problem-solving experience.