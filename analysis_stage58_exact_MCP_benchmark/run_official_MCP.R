args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) %in% c(0, 1))
out <- normalizePath(if (length(args)) args[1] else getwd(), winslash="/", mustWork=TRUE)
audit <- file.path(dirname(out), "analysis_stage57_external_resource_audit")
source(file.path(audit, "raw", "MCPcounter.R"), encoding="UTF-8")
genes <- read.delim(file.path(audit, "raw", "MCPcounter_genes.txt"), check.names=FALSE,
                   stringsAsFactors=FALSE, colClasses="character")
for (scope in c("source", "target")) {
  x <- as.matrix(read.csv(file.path(out, paste0(scope, "_official_marker_logTPM.csv")),
                         row.names=1, check.names=FALSE))
  storage.mode(x) <- "double"
  stopifnot(!anyNA(x), !anyDuplicated(rownames(x)))
  # Official function unchanged; locally supplied genes prevent its default network fetch.
  result <- MCPcounter.estimate(expression=x, featuresType="HUGO_symbols",
                               genes=genes, probesets=data.frame())
  stopifnot(nrow(result) == 10, ncol(result) == ncol(x))
  write.csv(result, file.path(out, paste0(scope, "_official_MCPcounter_scores.csv")),
            row.names=TRUE, fileEncoding="UTF-8")
  cat(scope, "official scores:", nrow(result), "populations;", ncol(result), "samples\n")
}
writeLines(capture.output(sessionInfo()), file.path(out, "official_R_sessionInfo.txt"))
