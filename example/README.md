# Single-subject HCP example

This example is configured for `100307`, the 3T MRI example subject in the WU-Minn Human Connectome Project (HCP) Young Adult S1200 release manual. The repository provides the subject list and input layout. Obtain the actual scans through HCP after accepting its Open Access Data Use Terms. [S1200 data release](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release/) · [HCP file layout for subject 100307](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Appendix_III%2B7T%2BDiffBed.pdf) · [HCP access and data use terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms)

1. Download the preprocessed 3T diffusion and structural MRI data for `100307` through the HCP portal. Prepare the HUMMID inputs and reference volumes listed in [input/](input/README.md).
2. Put your trained `HCP_WNet_T1w.pth` and `HCP_WNet_DWI.pth` checkpoints in `checkpoints/hummid/`. [HUMMID training instructions](../README.md#train-and-run-inference)
3. Install the [repository dependencies](../README.md#setup), then run from the repository root.

```bash
python -m inference.Inference --data-root example/input --subjects example/input/subjects.txt --model-dir checkpoints/hummid --output-dir example/output
```

The predictions will appear in [output/](output/README.md). The example demonstrates inference for a single subject. Training uses separate training and validation cohorts.

## Citations and acknowledgment

- Van Essen DC, Smith SM, Barch DM, Behrens TEJ, Yacoub E, Ugurbil K, for the WU-Minn HCP Consortium. The WU-Minn Human Connectome Project: An overview. *NeuroImage*. 2013;80:62–79. [doi:10.1016/j.neuroimage.2013.05.041](https://doi.org/10.1016/j.neuroimage.2013.05.041)
- Glasser MF et al. The minimal preprocessing pipelines for the Human Connectome Project. *NeuroImage*. 2013;80:105–124. [doi:10.1016/j.neuroimage.2013.04.127](https://doi.org/10.1016/j.neuroimage.2013.04.127)

We acknowledge the WU-Minn HCP Consortium, led by David Van Essen and Kamil Ugurbil, as the source of the example data. The consortium received support through NIH Blueprint for Neuroscience Research award 1U54MH091657 and the McDonnell Center for Systems Neuroscience at Washington University. [HCP citation and acknowledgment guidance](https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-citations)
