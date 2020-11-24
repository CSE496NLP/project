# EditNTS

This repository contains code primarily derived from
[EditNTS](https://github.com/yuedongP/EditNTS), as described in
"[EditNTS: An Neural Programmer-Interpreter Model for Sentence
Simplification through Explicit
Editing](https://arxiv.org/abs/1906.08104)".  The dataset handling
code has been modified to enable creation of SQLite databases, and
load datasets therefrom.

## Installation

This program is not prepared as a proper Python module.  However, a
Conda environment (`editnts-ppdb`) is provided in `environment.yml`,
and may be created with `conda env create --file=environment.yml`.

## Preprocessing of Data

To preprocess data, execute the `preprocess_main.py` script. The following
arguments can be used to specify the data to preprocess:

 - `--data-path`: This provides the path to the directory containing the dataset
   directories.
 - `--dataset`: This is the name of the directory within the directory specified
   by `--data-path` that contains the data you wish to preprocess.
 - `--file-base`: This is a prefix for the filenames within the directory
   specified by `--dataset` that you wish to preprocess. The files specified by
$FILEBASE.{src,dst} must exist within said directory.
 - `--ppdb_path`: The path to the directory containing PPDB files that you wish
   to preprocess. If this argument is not provided, no preprocessing will be
done with PPDB.
 - `--ppdb_version`: Name of the file within `--ppdb_path` to preprocess.

For wikilarge/small, a database will be created in `db/$FILEBASE.db`. For PPDB,
a database will be created in `db/ppdb.db`.

## Training & Testing

The following arguments are provided for managing execution of the
EditNTS model in training and testing scenarios.

The following provide paths to datasets:

 - `--data_set`: This provides the path to the testing data set.  It
   should be a `.db` or pickled Pandas frame.
 - `--eval_data_set`: This provides the path to the
   evaluation/validation dataset.
 - `--test_data_set`: This provides the path to the test data set.  It
   places the model in testing mode.  Only one of `--data_set` or
   `--test_data_set` may be provided.  Note that
   `--limit_test_elements` may also be provided, to use only the first
   $n$ elements from the test set and print the simplifications.

Vocabulary, model and storage:

 - `--vocab_path`: This is the path to the vocabulary data.
 - `--load_model`: This is the path to a checkpoint to load.
 - `--store_dir`: This is the path to a directory to store checkpoints
   in.
 - `--is_db`: If passed, datasets will be handled as SQLite databases.
 - `--max_seq_len`: Default 100.  This is the maximum length (in
   tokens) of a sentence to be simplified.
 - `--device`: Default 1.  This is the number of the GPU to use for
   training.

Hyperparameters:

 - `--vocab_size`: Default 30000.  The size of the vocabulary.
 - `--batch_size`: Default 32.  The batch size.
 - `--epochs`: Defau]lt 50.  The number of epochs to train.
 - `--hidden`: Default 200. The number of hidden units.
 - `--lr`: Default 1e-4.  This has no effect.
