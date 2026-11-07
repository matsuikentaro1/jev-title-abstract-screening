# Jev title and abstract screening: saved-output reanalysis

Reference labels, saved model predictions, prompts, settings, and analysis code for the study by Kentaro Matsui and Yoshikazu Takaesu.

Original titles and abstracts are not included: they contain third-party database content, including Embase, for which public redistribution permission has not been obtained. This package supports reanalysis of saved model outputs. It cannot rerun model screening without the original input texts and relevant API access.

## Recalculate the results

With Python 3.10 or later:

    python scripts/recalculate.py

The numerical analysis uses the Python standard library and requires no network connection or API key. It writes summary.json, tables_1_2.csv, figure_2_data.csv, and figure_3_data.csv to results/. Use --output-dir PATH to change the output folder.

Optional rendering of Figures 2 and 3 requires matplotlib:

    python -m pip install -r requirements-figures.txt
    python scripts/recalculate.py --figures

The graphs reproduce the numerical data and error ranges; their typography may differ from the manuscript. Figure 1 is a conceptual diagram.

## Calibration, confidence intervals, and supplementary analyses

These analyses require numpy and scipy:

    python -m pip install -r requirements-additional.txt
    python scripts/additional_analyses.py

The script writes CSV files to results/additional_analyses/ (change with --output-dir):

| File | Contents |
|---|---|
| 01a–01d | Analysis with the original labels before adjudication (Table S4) and retention of the reports ultimately included in the source reviews |
| 02a–02c | Calibration and discrimination of Jev probabilities (Brier score, expected calibration error, calibration slope, AUC); the calibration plot requires matplotlib |
| 03a–03b | Bootstrap 95% CIs and paired differences between Jev and each LLM condition (Table S2) |
| 04 | Cutoffs transferred between datasets |
| 05 | API charges and processing time per 1000 records |
| 06 | Human-retained records excluded by Jev Noul with one record per request or by Jev Choice in at least one run, as record identifiers (Table S1) |
| 07a–07c | Exclusion of human-retained records by position in the request and by number of records per request (Table S3) |

## Contents

| Path | Contents |
|---|---|
| data/records.csv | 4527 record identifiers, datasets, reference labels, source rows, missing-abstract flags, and input-text hashes |
| data/predictions.csv.gz | 267093 record-level predictions from 59 complete runs |
| data/requests.csv.gz | 138678 successful request-level observations, with each bundled request counted once |
| data/runs.json | Run conditions, repetitions, elapsed times, attempt counts, resumption flags, and API access dates |
| configuration/ | Prompts (prompts_noul.json for all Noul conditions; prompts_choice.json for Jev Choice and the LLMs), API settings, shuffle seeds, and reference-label provenance |
| scripts/recalculate.py | Reanalysis and optional figure generation |
| scripts/additional_analyses.py | Calibration, confidence intervals, and supplementary analyses |
| configuration/included_reports.json | Records corresponding to the reports ultimately included in the source reviews (identifiers only) |
| api_runners/, 1_data/README.md | API code and input specification retained to document model screening; the required input file is not supplied |
| environment.json | Verified reanalysis environment and available historical environment information |

The saved runs cover Jev noul-k1, noul-k10, noul-k25, noul-k50, and choice-v4 (10 runs each), and GPT-5 mini with minimal/default reasoning and GPT-6 Astra with default reasoning (3 runs each).

## Data and analysis notes

The datasets were reused from Matsui et al. ([2024](https://doi.org/10.2196/52758)) and originated from the reviews by [Takeshima et al.](https://doi.org/10.1111/pcn.12976) and [Maruki et al.](https://doi.org/10.1186/s40345-022-00271-7). The reference standard is retention for full-text assessment: 26 of 1381 light-therapy records (takeshima) and 87 of 3146 adjunctive-pharmacotherapy records (maruki). Original workbooks and individual reviewers' ratings are excluded.

Join predictions to requests using both run_id and request_id, and to records using record_id. Request identifiers are unique within each run. reference_retained and abstract_missing are 0/1 flags. probability_retained ranges from 0 to 1 for Jev and is empty for LLMs; decision stores the saved decision. Timestamps are UTC, latency is in milliseconds, and prices are USD per million tokens.

Record identifiers and source rows are internal references, not DOI/PMID retrieval identifiers. source_row is the one-based worksheet row, including the header. Input hashes are SHA-256 of UTF-8 text formed as "Title: " + title + newline + "Abstract: " + abstract, without normalization; a missing abstract is an empty string. These fields document provenance and cannot reconstruct the original texts. All screening records were included without additional deduplication.

At the fixed cutoff, Jev retains records with probability at least 0.50; saved LLM decisions are I or E. The highest cutoff retaining all reference-positive records in every run is selected on these evaluation data and is exploratory. The script also calculates cutoff transfer between datasets; this is not independent external validation.

Run means and ranges use unrounded values. Manuscript percentages and retained-count means are rounded to one decimal. Latency percentiles use the index floor(q*(n-1)); within-record probability variability uses population standard deviation across runs. Per-run confusion counts are also supplied.

Wall time is measured per run, includes active-run pacing and retries, and cannot be reconstructed by summing request latencies. Interruption time is excluded for one resumed Choice run. Estimated costs use successful-response tokens at study list prices, including LLM reasoning tokens in output tokens; setup calls and unobserved failed-call usage are excluded. Jev costs use input tokens only.

## Licenses

Code: [MIT](LICENSE). Author-generated reference labels and model-output data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These licenses do not cover the excluded third-party titles and abstracts.
