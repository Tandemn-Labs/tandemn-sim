# Three-hour simulation trace analyses

This directory stores analysis JSON derived from production datasets in the
sibling `raw_datasets` directory. Normalized datasets and selected dataset
slices do not belong here.

Each analysis must be registered in `../TRACE_INDEX.md` with:

- the source dataset file;
- applied filters;
- selected time window, when applicable;
- traffic and token metrics.

The 18 lowercase `*_analysis.json` files are the job-specific analyses for the
online benchmark jobs. They are generated from `../trace_mapping.yaml` by the
shared `data_generation.benchmark_trace_analysis` runner.

`BurstGPT_3_analysis.json` is an exploratory analysis of the complete BurstGPT
source file. It is not yet a job-specific analysis.
