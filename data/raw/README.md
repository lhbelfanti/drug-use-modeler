# Raw Data Documentation

This directory contains the source CSV files for the Drug Use Modeler project.

## Datasets

There are two primary dataset variations:

1.  **`raw-corpus.csv`**: The full, unfiltered dataset.
2.  **`pre-filtered-corpus.csv`**: A filtered version of the dataset (though currently matches the raw corpus in size).

## Sample Counts

Both datasets currently contain exactly **3,000 samples** each.

| Dataset | Total Samples | Positive Class | Negative Class |
| :--- | :---: | :---: | :---: |
| **Raw Corpus** | 3,000 | 1,500 | 1,500 |
| **Pre-filtered Corpus** | 3,000 | 1,500 | 1,500 |

## Substance Breakdown

Both CSVs now include a `SearchCriteriaID` column (added in `corpus-creator`, mapped to the search criteria that originated each tweet), which lets the train/val/test split be stratified by substance in addition to class.

| Substance | `SearchCriteriaID` | Total Samples | Positive Class | Negative Class |
| :--- | :---: | :---: | :---: | :---: |
| **Cocaína** | 1, 2, 3, 4 | 1,000 | 500 | 500 |
| **Marihuana** | 5 | 1,000 | 500 | 500 |
| **Heroína** | 6 | 752 | 376 | 376 |
| **Ecstasy** | 7 | 248 | 124 | 124 |

See [`notebooks/02_preprocessing.ipynb`](../../notebooks/02_preprocessing.ipynb) for the split logic (70/15/15, stratified jointly by class and substance). The notebook also resolves cross-split text duplicates after the split (`resolve_cross_split_duplicates`) — several tweets in the corpus are exact-text duplicates independently posted by different users, which a plain stratified split can scatter across train/val/test; these are now consolidated into a single split via same-label/same-drug swaps, so split sizes and balance are unaffected.

> **Note**: `pre-filtered-corpus.csv` previously had a text-corruption bug inherited from `corpus-creator`'s cleaning rules (e.g. `más` → `meás`, `día` → `deía`), affecting ~350/3,000 rows. Fixed at the source (see [`corpus-creator` README](https://github.com/lhbelfanti/corpus-creator#readme)) and the corpus was regenerated — this file now reflects the corrected text. See [Iteration 3](../../evaluation/README.md#iteration-3-honest-beto-selection--deduplicated-split) in the evaluation log.

