# Publication figures

These scripts render the final six main and two supplementary figures from the separate scientific resource. They contain no input tables or plotted assets. Figure-specific assertions retain the frozen panel populations.

Install the plotting requirements, then run:

```shell
python tools/figures/render_all.py --audit-root EXTERNAL_RESOURCE/audit --output NEW_FIGURE_DIRECTORY
```

The input tree uses the source-relative layout in the separate resource. Some upstream arrays must be obtained from the original providers listed in DATA_ACCESS.md. Output must be outside that original tree. Arial must be installed; alternatively supply --font-regular and --font-bold with licensed Arial font files. Individual builders accept the same options. Each family runs in a separate process to avoid global plotting-setting leakage.

The portable version changes path configuration and code formatting only. It is checked against all eight already reviewed final PNG files; see PLOTTING_QA.json in the repository root.
