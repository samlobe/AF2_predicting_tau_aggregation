#%%
import numpy as np
import pandas as pd

# read the sequence full_tau.fasta file
full_tau_fasta = open('full_tau.fasta', 'r')
full_tau_fasta = full_tau_fasta.readlines()
full_tau_sequence = ''.join(line.strip() for line in full_tau_fasta[1:])

# read the ThT csv file to get resid numbers
tht_df = pd.read_csv('ThT_PAM4_paper.csv',index_col=0)
peptides = tht_df.columns
start_resids = [int(peptide.split('-')[0]) for peptide in peptides[:-1]] # ignoring the control column
end_resids = [int(peptide.split('-')[1]) for peptide in peptides[:-1]]

# check that all the lengths are 15
lens = [end_resids[i] - start_resids[i] + 1 for i in range(len(start_resids))]
if not all([l == 15 for l in lens]):
    print('Error: not all peptides are 15 residues long')
    print(lens)
    print(peptides)
    exit()

# %%
# loop through the start ids to get all the sequences
seqs = []
for i in range(len(start_resids)):
    seqs.append(full_tau_sequence[start_resids[i]-1:end_resids[i]])

#%%
# how many peptides to put in each multimer
n_peptides = 5    

# write the sequences to a new fasta file
with open('multimer.fasta', 'w') as f:
    for seq, peptide in zip(seqs, peptides[:-1]):
        f.write('>' + peptide + '\n')
        for i in range(n_peptides-1):
            f.write(seq + ':\n')
        f.write(seq + '\n')

#%%
# do the same thing except for regular AF2 with 5 U linker
with open('5U.fasta', 'w') as f:
    for seq, peptide in zip(seqs, peptides[:-1]):
        f.write('>' + peptide + '\n')
        for i in range(n_peptides-1):
            f.write(seq + 'UUUUU' + '\n')
        f.write(seq + '\n')

# and for 10 U linker
with open('10U.fasta', 'w') as f:
    for seq, peptide in zip(seqs, peptides[:-1]):
        f.write('>' + peptide + '\n')
        for i in range(n_peptides-1):
            f.write(seq + 'UUUUUUUUUU' + '\n')
        f.write(seq + '\n')

#%%
# write esm fasta with each sequence (no repeats)
with open('esm.fasta', 'w') as f:
    for seq, peptide in zip(seqs, peptides[:-1]):
        f.write('>' + peptide + '\n')
        f.write(seq + '\n')