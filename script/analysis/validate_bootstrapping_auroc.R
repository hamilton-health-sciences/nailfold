library(dplyr)
library(cvAUC)

opts <- commandArgs(trailingOnly=TRUE)
if (length(opts) < 1) {
    message("Please provide a path to the predictions file.")
    quit()
}
pred_fn <- opts[1]

sigmoid <- function(x) 1/(1 + exp(-x))

preds.img <- read.csv(pred_fn)
preds.patient <- preds.img %>%
                 group_by(id) %>%
                 summarize(
                     yprob=mean(sigmoid(yhat)),
                     y=max(y),  # cheap way to carry over, all should be id
                     fold=min(fold)
                 )
preds.patient <- as.data.frame(preds.patient)

results.img <- ci.cvAUC(
    preds.img$yhat, preds.img$y, folds=preds.img$fold
)
results.patient <- ci.cvAUC(
    preds.patient$yprob, preds.patient$y, folds=preds.patient$fold
)

message("Image-level results:")
message("  AUROC: ", results.img$cvAUC, " (", results.img$confidence,
        " pct CI: ", results.img$ci[1], " - ", results.img$ci[2], ")")

message("Patient-level results:")
message("  AUROC: ", results.patient$cvAUC, " (", results.patient$confidence,
        " pct CI: ", results.patient$ci[1], " - ", results.patient$ci[2], ")")
