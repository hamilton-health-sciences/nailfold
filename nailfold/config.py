"""Global configuration options used end-to-end."""

# The root of the dataset. Should contain patient images, image metadata, and
# outcome data.
DATA_ROOT = ""
RAY_LOGS_ROOT = ""
RESULTS_ROOT = ""

# Split information. Defaults are 5-fold stratified cross-validation, with 20%
# of the training set being split off for hyperparameter tuning.
NUM_FOLDS = 5
VAL_SET_PROPORTION = 0.2

# The input dimensions of the image ([height, width]).
INPUT_IMAGE_SIZE = (375, 500)

# Total number of epochs to train for.
NUM_EPOCHS = 100

# Minimum number of epochs to train for. This serves as the grace period for
# the hyperparameter search algo.
MIN_TRAINING_EPOCHS = 10

# Number of tuning fits for each outcome-fold.
NUM_TUNING_FITS_NONCONDITIONAL = 15
NUM_TUNING_FITS_CONDITIONAL = 30

# Computational limits.
NUM_DATALOADER_WORKERS = 8
MAX_BATCH_SIZE = 8
MAX_REPRESENTATION_BATCH_SIZE = 4096

# Seed and number of samples to use for bootstrapping confidence intervals.
BOOTSTRAP_SEED = 0
NUM_BOOTSTRAP_SAMPLES = 1000
