# The inference the README's section 2.2 argument rests on, redone in base R.
#
# Two claims there are statistical rather than arithmetic, and neither is
# recomputed anywhere else in this repository:
#
#   1. the naive label-agreement measure tracks edge-reading ability at
#      r = -0.924 while the within-node measure does not (r = +0.004), which is
#      the whole reason the competence-floor reading was withdrawn
#   2. two of the five models read a six-node edge list at chance
#
# The first is redone with an exact permutation test over all 120 relabellings
# rather than a t approximation, which is the only honest test at n = 5. The
# second is redone with an exact binomial test, a different route to the same
# conclusion the Wilson intervals in reports/edge_reading_control.csv reach.
#
# Base R only, so CI needs nothing installed beyond r-base-core.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

failures <- 0
fail <- function(...) {
    cat("FAIL", sprintf(...), "\n")
    failures <<- failures + 1
}

joint <- read.csv(file.path(root, "reports", "competence_vs_label.csv"),
                  check.names = FALSE)
# One arm per model. The saliency arm exists for one model only, so including it
# would weight that model twice in a correlation over five points.
joint <- joint[joint$explainer == "gnnexplainer", ]
cat(sprintf("%d models, gnnexplainer arm\n", nrow(joint)))
if (nrow(joint) != 5) {
    fail("expected 5 models in the gnnexplainer arm, found %d", nrow(joint))
}

# All permutations of 1..n. n is 5 here, so 120 of them: the permutation
# distribution is enumerated rather than sampled and the p value is exact.
perms <- function(v) {
    if (length(v) <= 1) return(list(v))
    out <- list()
    for (i in seq_along(v))
        for (rest in perms(v[-i])) out[[length(out) + 1]] <- c(v[i], rest)
    out
}

permutation_p <- function(x, y) {
    r0 <- cor(x, y)
    all <- vapply(perms(seq_along(y)), function(p) cor(x, y[p]), numeric(1))
    list(r = r0, p = mean(abs(all) >= abs(r0) - 1e-12), n = length(all))
}

readme <- paste(readLines(file.path(root, "README.md"), warn = FALSE),
                collapse = "\n")

check <- function(label, x, y, sign, want_r, want_p) {
    res <- permutation_p(x, y)
    cat(sprintf("  %-22s r %+.3f  exact p %.3f over %d permutations\n",
                label, res$r, res$p, res$n))
    if (sign * res$r <= 0) fail("%s: r has the wrong sign", label)
    if (abs(abs(res$r) - want_r) > 5e-4) {
        fail("%s: |r| %.4f against %.3f in the README", label, abs(res$r), want_r)
    }
    if (abs(res$p - want_p) > 5e-4) {
        fail("%s: p %.4f against %.3f in the README", label, res$p, want_p)
    }
    # And the README has to still say what R just computed.
    for (s in c(sprintf("%.3f", want_r), sprintf("%.3f", want_p))) {
        if (!grepl(s, readme, fixed = TRUE)) {
            fail("%s: the README no longer contains %s", label, s)
        }
    }
}

cat("\ncorrelation with edge-reading ability, exact permutation test\n")
check("naive label agr.", joint$edge_reading, joint$follows_label, -1, 0.924, 0.058)
check("label sensitivity", joint$edge_reading, joint$label_sensitivity, +1, 0.004, 0.992)

# The two measures disagree about the competence floor, which is the point of
# section 2.2. If they ever agreed, the section would be wrong.
if (sign(cor(joint$edge_reading, joint$follows_label)) ==
    sign(cor(joint$edge_reading, joint$label_sensitivity))) {
    fail("the two measures now point the same way")
}

# Exact binomial test on the control arm, against the Wilson intervals the
# published table reports. Two of five must fail to reject chance.
cat("\ncontrol arm against chance, exact binomial test\n")
ctl <- read.csv(file.path(root, "reports", "edge_reading_control.csv"))
ctl <- ctl[ctl$explainer == "gnnexplainer", ]
p_hat <- as.numeric(sub(" .*", "", ctl$edge_reading_accuracy))
lo <- as.numeric(sub(".*\\[([0-9.]+),.*", "\\1", ctl$edge_reading_accuracy))
hi <- as.numeric(sub(".*,([0-9.]+)\\]", "\\1", ctl$edge_reading_accuracy))
at_chance <- 0
for (i in seq_len(nrow(ctl))) {
    k <- round(p_hat[i] * ctl$n[i])
    bt <- binom.test(k, ctl$n[i], 0.5)
    covers <- lo[i] <= 0.5 && 0.5 <= hi[i]
    if (bt$p.value > 0.05) at_chance <- at_chance + 1
    # The exact test and the published interval have to agree about this model,
    # or one of the two is wrong.
    if ((bt$p.value > 0.05) != covers) {
        fail("%s: binomial p %.3f but the published interval %s 0.5",
             ctl$model[i], bt$p.value, if (covers) "covers" else "excludes")
    }
    cat(sprintf("  %-24s %2d/%3d  p %.4f  published interval %s 0.5\n",
                ctl$model[i], k, ctl$n[i], bt$p.value,
                if (covers) "contains" else "excludes"))
}
cat(sprintf("  %d of %d models are indistinguishable from chance\n",
            at_chance, nrow(ctl)))
if (at_chance != 2) {
    fail("the README says two of five models read at chance, R finds %d", at_chance)
}

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces both correlations, both exact p values, and the two",
    "\nmodels that read the edge list at chance\n")
