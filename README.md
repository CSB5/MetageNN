# MetageNN
MetageNN is a proof of concept memory-efficient long-read taxonomic classifier that is robust to sequencing errors and missing genomes. MetageNN is based on a neural network model that uses short k-mer profiles of sequences to reduce the impact of “distribution shift” when extrapolating from training on genome sequences to testing on error-prone long reads. MetageNN can be used on sequences not classified by conventional methods and offers an alternative approach for memory-efficient classifiers that can be optimized further.

## Requirements
```
bash install_requirements.sh
```

## Data
You can find the list of genomes used to train (either small or the main database) as well as the list of isolates used to test at /data. A link is also provided to download the "small database" training dataset (1x coverage) that can be used to train MetageNN.

## Counting k-mers

MetageNN can be trained using any sequence length. For our proof of concept, we sampled sequences of 1kbp from genomes. To count the k-mers of these sequences, we used the Phylopythia k-mer counting algorithm [1] by using the following command:

```
save_file = 'path_to_file/genome_segments.fasta'
fasta2kmers2 -i save_file -j 6 -k 6 -s 0 -l 0 -n 1 -f k_mers_counted.6mer        
```
In the example above, we counted canonical 6mers given an input file in fasta format.

## Settings

In the /settings folder you will find JSON files containing the best hyperparameters for MetageNN for both databases (the small database and the main database). MetageNN can load these files during training.

## Training

To train MetageNN on the small database of genomes you can run (please download the "small database" training dataset first found at /data):

```
python code/MetageNN_train.py -s settings/MetageNN_settings_small_database.json
```

## Cite

Preprint to be released.

## References

[1] https://github.com/algbioi/kmer_counting 

## Contact
For additional information, help and bug reports please email Rafael Peres da Silva (rperesdasilva@gis.a-star.edu.sg).
