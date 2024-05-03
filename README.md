# AF2_predicting_tau_aggregation
Trying to see if any AF2 metric can predict which tau 15mers aggregate according to ThT or pFTAA assays from [PAM4 paper](https://www.nature.com/articles/s41467-024-45429-2/figures/1).
Also sharing part of my AF2 analysis workflow for collaborators.

Workflow:
1) Create the .fasta files using `generate_fastas.py`
   - I test out a) the multimer model, b) using 5Us between chains, and c) using 10Us between chains. Each with 5 chains.
2) Run the AF2 predictions on Pod, taking the fastas as input. You can see my submission scripts in the `submission_scripts` directory. You can see my AF2 tutorial for Pod [here](https://roamresearch.com/#/app/SamLobo/page/oF3yTZG6x). Copy the results back here.
   - Note: I test out multimer, multimer with max 5 recycles (like in the [LIS score paper](https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1.full.pdf)), 5U (not multimer), and 10U (not multimer).
3) Measure the alpha carbon pairwise distances between each residues using `pairdist_map.py`. This is useful for two of the AF2 metrics that I measure for Step 4.
4) Measure the [partial] intermolecular hbonds using dssp.py
5) Measure several AF2 metrics using `compile_scores.py` to create a "features.csv" file.
6) See if individual features can discern aggregators from non-aggregators using `classify.py`
   - I did some exploratory data analysis in `EDA.py`.
   - `EDA.py` and `true_labels.py` both output a binary array to denote aggregators vs non-aggregators. They read some csv files ("THT_PAM4_paper.csv" and "pFTAA_PAM4_paper.csv" that I got from the PAM4 paper.)

