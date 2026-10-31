# Data-driven groups of nationalities: hierarchical clustering and PCA of the nationality profiles,
# with a bootstrap of the answers to measure how stable the groups are. Sourced by report.qmd.

# Indicators of a profile (share of answers, both models pooled).
FEAT <- c(defensive_marking = "\"origin does not matter\"", stereo_pos = "positive stereotype",
          stereo_neg = "negative stereotype", negative_association = "negative association",
          alarm_high = "high alarm (≥ 3)", caution_alarm = "caution or alarm framing",
          protective = "protective advice", encounter = "encounter framing",
          asks_clarification = "asks for clarification")

# Smoothed share (k + 0.5) / (n + 1), so that rare indicators at 0 stay finite on the logit scale.
profiles <- function(data, by = "country") {
  data |> group_by(across(all_of(by))) |>
    summarise(n = n(), across(all_of(names(FEAT)), ~ (sum(.x, na.rm = TRUE) + 0.5) / (sum(!is.na(.x)) + 1)),
              .groups = "drop")
}

# Standardised logits: each indicator weighs the same whatever its base rate.
zmat <- function(P) {
  X <- as.matrix(P[, names(FEAT)])
  rownames(X) <- P$country
  scale(qlogis(X))
}

cluster_of <- function(P, k) cutree(hclust(dist(zmat(P)), "ward.D2"), k)

# Adjusted Rand index between two partitions.
ari <- function(a, b) {
  t <- table(a, b); s <- function(x) sum(choose(x, 2))
  e <- s(rowSums(t)) * s(colSums(t)) / choose(sum(t), 2)
  (s(t) - e) / ((s(rowSums(t)) + s(colSums(t))) / 2 - e)
}

# Bootstrap: resample the answers within each nationality, recompute the profiles and the clustering, and
# match each reference group to its closest bootstrap group (Jaccard), as in Hennig (2007).
bootstrap_clusters <- function(N, k_range = 2:6, B = 500, seed = 1) {
  set.seed(seed)
  idx <- split(seq_len(nrow(N)), N$country)
  P <- profiles(N)
  boots <- replicate(B, profiles(N[unlist(lapply(idx, \(i) sample(i, replace = TRUE))), ]), simplify = FALSE)
  per_k <- lapply(k_range, function(k) {
    ref <- cluster_of(P, k)
    res <- lapply(boots, function(bp) {
      b <- cluster_of(bp, k)
      jac <- sapply(seq_len(k), \(c) sapply(seq_len(k), \(d) sum(ref == c & b == d) / sum(ref == c | b == d)))
      best <- apply(jac, 2, which.max)  # bootstrap group matched to each reference group
      list(jac = apply(jac, 2, max), same = b == best[ref], ari = ari(ref, b))
    })
    list(k = k, ref = ref,
         jaccard = rowMeans(sapply(res, `[[`, "jac")),
         stay = setNames(rowMeans(sapply(res, `[[`, "same")), P$country),
         ari = mean(sapply(res, `[[`, "ari")))
  })
  setNames(per_k, k_range)
}

# Segments of a dendrogram (leaves at y = 1..n in the order of the tree, height on x).
dendro_segments <- function(h) {
  n <- length(h$order)
  pos <- numeric(n); pos[h$order] <- seq_len(n)
  node_y <- numeric(n - 1); node_x <- h$height
  y_of <- function(i) if (i < 0) pos[-i] else node_y[i]
  x_of <- function(i) if (i < 0) 0 else node_x[i]
  segs <- vector("list", n - 1)
  for (m in seq_len(n - 1)) {
    a <- h$merge[m, 1]; b <- h$merge[m, 2]
    node_y[m] <- (y_of(a) + y_of(b)) / 2
    segs[[m]] <- data.frame(x = c(x_of(a), x_of(b), node_x[m]), xend = rep(node_x[m], 3),
                            y = c(y_of(a), y_of(b), y_of(a)), yend = c(y_of(a), y_of(b), y_of(b)))
  }
  do.call(rbind, segs)
}
