Replication
===========

The results in the paper are end-to-end reproducible. The main software dependency is Python. The code assumes GPUs are available.

Requirements
------------

Under Python 3.6.9, run:

    $ python3 -m pip install -r requirements.txt

This will install the required software.

Preprocessing
-------------

    $ mkdir processed
    $ python3 script/process.py --data_root ./data --output_root ./processed

This will create a new directory structure which mirrors the original. In
particular, the patient directories will be re-created, and the unannotated
images will be symlinked to the original paths. Additionally, image metadata
(including measurement annotations from the corresponding structured text files,
if available) and outcome data will be procssed into CSV files, with the outcome
data coming from the original Excel file.

There are two image directories labelled patient 38. However, one is very
obviously patient 37. Therefore after running `process.py`, one should rename
directory `processed/38_30-12-2019_14.52.41` to
`processed/37_30-12-2019_14.52.41` and conduct an identical find-and-replace
in `processed/metadata.csv`:

    $ cd processed && mv metadata.csv metadata_orig.csv
    $ cat metadata_orig.csv | \
    > sed 's/38_30-12-2019_14.52.41/37_30-12-2019_14.52.41/' | \
    > awk -F',' '{if ($5 !~ /37_30-12-2019_14.52.41/) { print($0) } else { printf 37 ","; for (i = 2; i < NF; i++) printf $i ","; print $NF;}}' \
    > > metadata.csv
    $ mv "38_30-12-2019_14.52.41" "37_30-12-2019_14.52.41"

Configuration
-------------

`nailfold/config.py` contains some high-level parameters that control the pipeline.
Almost all of them should not be tweaked in the interest of reproducibility. The
exceptions are the filepaths at the top of the file, which should be re-
configured accordingly.

Modeling
--------

    $ bash script/prediction/run_tune.sh $outcome $repetition

This will run the tuning procedure for the outcome `$outcome` (which should be
one of `diabetes`, `hba1c_high`, `cardiovascular_event`, `hypertension`,
`retinopathy`) for all 5 folds within a given repetition (`$repetition = 0` for
the results given).

For conditional outcomes, the form looks like:

    $ bash script/prediction/run_tune.sh $outcome $conditional $repetition

where for this study we consider outcomes conditional on diabetes, so
`$conditional = diabetes`.

This will generate `ray_logs` (and possibly `lightning_logs`) directories.

Evaluation
----------

    $ bash script/analysis/evaluate.sh $outcome $repetition

or for conditional outcomes

    $ bash script/analysis/evaluate.sh $outcome $conditional $repetition

where `$conditional` is as defined above.

This will generate evaluation metrics, saliency maps, and perform measurement
coherence tests for a given outcome. The output will be placed in
`results/$outcome` or `results/${outcome}_cond_${conditional}`.

# Validation

Unit tests are used to validate that the study design and the above
reproducibility controls are implemented properly and consistently. These are
available in `code/test`. They can be run like so:

    $ cd code
    $ PYTHONPATH=. pytest --mypy
