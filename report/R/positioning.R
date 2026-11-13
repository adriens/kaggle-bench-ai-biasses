# Data and charts of the positioning documents (positioning.qmd and positioning_short.qmd): the RTS article,
# UN report A/HRC/56/68 and the full report, side by side. Sourced after R/prep.R.

# one colour per work, checked for colour-blind separation; the labels carry the identity too
STUDY <- c(RTS = "#168F63", UN = "#2A78D6", "This study" = "#D63C2A")
INK <- "#1b1b1b"; MUTED <- "#5b5a56"; GRID <- "#e4e3df"

theme_set(
  theme_minimal(base_size = 9.5, base_family = "Inter") +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = GRID, linewidth = 0.3),
          axis.text = element_text(colour = MUTED),
          axis.title = element_text(colour = MUTED),
          plot.title.position = "plot",
          legend.position = "top",
          legend.title = element_blank(),
          strip.text = element_text(face = "bold", hjust = 0))
)
num <- function(x, d = 1) formatC(x, format = "f", digits = d, big.mark = ",")
pc <- function(x, d = 0) paste0(num(x, d), " %")
# every table gets a bold header row: kable writes a Markdown (pipe) table, whose first line is the header
kable <- function(...) {
  k <- knitr::kable(...)
  if (!identical(attr(k, "format"), "pipe")) return(k)
  cells <- strsplit(k[1], "|", fixed = TRUE)[[1]]
  bold <- function(x) {
    t <- trimws(x)
    if (t == "" || startsWith(t, "**")) x else sub(t, paste0("**", t, "**"), x, fixed = TRUE)
  }
  k[1] <- paste(vapply(cells, bold, ""), collapse = "|")
  if (!endsWith(k[1], "|")) k[1] <- paste0(k[1], "|")
  k
}

# the key figures, computed as in the short version
SIZE_B <- c("granite-4.2-8b" = 8, "qwen3.5-9b" = 9, "gemma-4-e2b-it" = 2, "gemma-4-e4b-it" = 4,
            "gemma-4-26b-a4b-it-awq-4bit" = 26, "ministral-3-8b-instruct-2512" = 8,
            "nvidia-nemotron-3.5-bolt-e4b-bf16" = 4)
TD <- D |> mutate(prej = negative_association | defensive_marking | coalesce(stereo_neg, FALSE))
risky <- TD |> filter(SIZE_B[model] >= 6, context_id %in% c("C3_taxi", "C4_street", "C9_elevator"))
p_none <- 100 * mean(risky$prej[risky$is_base]); p_nat <- 100 * mean(risky$prej[!risky$is_base])
gm_ <- read.csv(file.path("..", "analysis", "out", "guess", "metrics.csv"))
g_acc <- 100 * gm_$accuracy[gm_$setting == "shared" & gm_$scope == "both" & gm_$algo == "logreg"]
w_rho <- read.csv(file.path("..", "analysis", "out", "wealth_spearman.csv"))
rho_c <- w_rho$rho[w_rho$measure == "compliment"]
n_countries <- n_distinct(D$country[!D$is_base])

LEVELS <- c(m = "Measured", i = "Illustrated", c = "Stated", a = "Not covered")
cov <- tibble::tribble(
  ~question, ~ref, ~RTS, ~UN, ~study, ~report,
  "Technology is not neutral", "UN §7–8, §66", "i", "c", "m", "Sections 3.1, 4",
  "Do AIs reproduce learned stereotypes?", "UN §13–15", "i", "c", "m", "Sections 3.2, 3.4, 3.13",
  "Design choices and proxies", "UN §17–18", "c", "c", "i", "Section 3.7",
  "National origin as a ground", "ICERD art. 1; UN §12, §59", "i", "c", "m", "Section 2.1",
  "Generative AI and large language models", "UN §11, §47", "i", "c", "m", "Section 2.2",
  "A language other than English", "UN §47", "i", "c", "m", "Sections 2.1, 5",
  "All else being equal (a control group)", "method", "a", "a", "m", "Sections 2.1, 3.1",
  "Age, minors included", "UN §26", "a", "c", "m", "Section 3.9",
  "Global North and South", "UN §58, §68 n", "a", "c", "m", "Section 3.11",
  "Testing before deployment, with thresholds", "UN §68 d", "a", "c", "m", "Sections 2.1–2.5",
  "Independent audit, monitoring over time", "UN §68 m, §69 d", "a", "c", "m", "Section 2.2; Data and code availability",
  "Transparency, out of the black box", "UN §21–23, §68 f", "a", "c", "m", "Section 2.4; Data and code availability",
  "Rejecting a \"colour-blind\" approach", "UN §66", "a", "c", "m", "Section 3.3",
  "Public AI literacy", "UN §67, §68 o, §69 f", "a", "c", "i", "TL;DR; short version and carousel",
  "Personal safety advice", "outside the UN domains, §25", "i", "a", "m", "Sections 2.1, 3.8"
) |> rename(`This study` = study)
cov_long <- cov |>
  pivot_longer(c(RTS, UN, `This study`), names_to = "work", values_to = "lvl") |>
  mutate(work = factor(work, levels = names(STUDY)),
         level = factor(LEVELS[lvl], levels = LEVELS),
         label = paste0(question, "  (", ref, ")"),
         label = factor(label, levels = rev(paste0(cov$question, "  (", cov$ref, ")"))))

SHAPES <- c("Measured" = 16, "Illustrated" = 10, "Stated" = 1, "Not covered" = 95)
cover_plot <- ggplot(cov_long, aes(work, label)) +
  geom_point(aes(shape = level, colour = work), size = 4.6, stroke = 1) +
  scale_shape_manual(values = SHAPES, drop = FALSE) +
  scale_colour_manual(values = STUDY, guide = "none") +
  scale_x_discrete(position = "top") +
  labs(x = NULL, y = NULL) +
  theme(panel.grid.major.x = element_blank(),
        axis.text.x = element_text(colour = INK, face = "bold", size = 10),
        axis.text.y = element_text(colour = INK),
        legend.location = "plot", legend.justification = "left", legend.text = element_text(size = 8.5, colour = INK),
        legend.margin = margin(0, 0, 0, 0))
tally <- cov_long |>
  count(work, level, .drop = FALSE) |>
  mutate(level = factor(level, levels = rev(LEVELS))) |>
  ggplot(aes(n, work, fill = work, alpha = level)) +
  geom_col(colour = "white", linewidth = 0.8, width = 0.7) +
  # the level is named only where the segment is wide enough; the legend above names the others
  geom_text(aes(label = ifelse(n >= 4, paste(n, tolower(level)), ifelse(n > 0, n, ""))),
            position = position_stack(vjust = 0.5), colour = INK, size = 3.4, alpha = 1, family = "Inter") +
  scale_fill_manual(values = STUDY, guide = "none") +
  scale_alpha_manual(values = c("Measured" = 0.9, "Illustrated" = 0.55, "Stated" = 0.3, "Not covered" = 0.08), guide = "none") +
  scale_x_continuous(breaks = c(0, 15), expand = expansion(mult = c(0, 0.02))) +
  scale_y_discrete(limits = rev(names(STUDY))) +
  labs(x = "questions", y = NULL) +
  theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(colour = INK, face = "bold", size = 10.5))

# each item: the part of the UN report that raises it, its short name, the level of the answer of this study
# and the part of the full report that answers it (the first section of Table 2)
UN_PART <- c(F = "Framework\n§7–12", M = "Mechanisms\n§13–23", D = "Domains\n§24–50", R = "Regulation\n§51–58",
             L = "Law and\nrecommendations\n§59–70", X = "Not raised\nby the UN")
REP_PART <- c(meth = "Method\nSections 2.1–2.5", res = "Results\nSections 3.1–3.13",
              data = "Data and code\navailability", pub = "TL;DR, short\nversion, carousel")
items <- tibble::tribble(
  ~un, ~name, ~lvl, ~rep,
  "F", "Technology is not neutral", "m", "res",
  "F", "National origin as a ground", "m", "meth",
  "F", "Generative AI and LLMs", "m", "meth",
  "M", "Learned stereotypes", "m", "res",
  "M", "Design choices and proxies", "i", "res",
  "M", "Transparency, the black box", "m", "data",
  "M", "Privacy of the data", "o", NA,
  "M", "Intentional discriminatory use", "o", NA,
  "M", "Remedies and appeals", "o", NA,
  "D", "Non-English language", "m", "meth",
  "D", "Age, minors included", "m", "res",
  "D", "Sector cases (policing, health…)", "o", NA,
  "R", "North and South", "m", "res",
  "R", "Regulation, EU AI Act", "o", NA,
  "L", "Testing before deployment", "m", "meth",
  "L", "Audits over time", "m", "data",
  "L", "Rejecting colour-blindness", "m", "res",
  "L", "Public AI literacy", "i", "pub",
  "L", "Binding standards, bans", "o", NA,
  "L", "Consultation, training", "o", NA,
  "X", "A control group", "m", "meth",
  "X", "Personal safety advice", "m", "meth"
) |>
  mutate(un = factor(un, levels = names(UN_PART)), rep = factor(rep, levels = names(REP_PART)),
         lvl = factor(lvl, levels = c("m", "i", "o"))) |>
  # within a part of the UN report: the answered questions first, in the order of their answer, then the others
  arrange(un, lvl == "o", rep) |>
  mutate(i = row_number())

# one fill scale for bands (b) and pills (p); bands carry their transparency in the colour
FILL <- c(m = "#D63C2A", i = "#E98A7B", o = "#B9B7B1")      # measured, illustrated, outside this study
FILLS <- c(setNames(paste0(FILL, "55"), paste0("b", names(FILL))),
           pm = "#F7DCD7", pi = "#FBEDEA", po = "#F1F0EC")
UNC <- "#2A78D6"; REPC <- "#D63C2A"; XC <- "#8D8B85"

T <- 0.78          # thickness of a band
STEP <- 1.32       # distance between two questions in the middle column
GGAP <- 0.75       # extra gap between two parts of the UN report
XL <- c(0, 0.035); XM <- c(0.36, 0.96); XR <- c(1.285, 1.32)

items <- items |> mutate(y = -((i - 1) * STEP + (as.integer(un) - 1) * GGAP))
# a block is centred on the questions it holds, then pushed down if it would overlap the block above
place <- function(df, key, gap = 1.1) {
  b <- df |> group_by(.data[[key]]) |> summarise(n = n(), c = mean(y), .groups = "drop") |>
    arrange(desc(c)) |> mutate(h = n * T, top = c + h / 2)
  for (k in seq_len(nrow(b))[-1]) b$top[k] <- min(b$top[k], b$top[k - 1] - b$h[k - 1] - gap)
  b |> mutate(bot = top - h)
}
un_blocks <- place(items, "un")
rep_blocks <- place(items |> filter(!is.na(rep)), "rep")

# where a band leaves or enters a block: stacked from the top, in the order of the middle column
items <- items |>
  group_by(un) |> mutate(l_top = un_blocks$top[match(un, un_blocks$un)] - (row_number() - 1) * T) |> ungroup() |>
  arrange(desc(y)) |>
  group_by(rep) |> mutate(r_top = rep_blocks$top[match(rep, rep_blocks$rep)] - (row_number() - 1) * T) |> ungroup()

# a band: two smoothstep curves between (x0, top0) and (x1, top1)
band <- function(id, x0, x1, top0, top1, lvl) {
  u <- seq(0, 1, length.out = 60); s <- u * u * (3 - 2 * u)
  x <- x0 + (x1 - x0) * u; top <- top0 + (top1 - top0) * s
  tibble::tibble(id = id, lvl = lvl, x = c(x, rev(x)), y = c(top, rev(top - T)))
}
left <- purrr::pmap_dfr(list(items$i, items$l_top, items$y, as.character(items$lvl)),
                        \(i, t0, ym, l) band(paste0("l", i), XL[2], XM[1], t0, ym + T / 2, l))
ans <- items |> filter(!is.na(rep))
right <- purrr::pmap_dfr(list(ans$i, ans$y, ans$r_top, as.character(ans$lvl)),
                         \(i, ym, t1, l) band(paste0("r", i), XM[2], XR[1], ym + T / 2, t1, l))


# the Sankey diagram, drawn for a figure of W_IN x H_IN inches (the rounded ends of the pills depend on it)
sankey_plot <- function(W_IN, H_IN) {
  # a pill: a rectangle with rounded ends, the radius converted from inches to each axis
  XLIM <- c(-0.42, 1.78); YLIM <- c(min(items$y) - 1.6, 2.6)
  rx <- 0.09 * diff(XLIM) / W_IN; ry <- 0.09 * diff(YLIM) / H_IN
  pill <- function(id, x0, x1, yc, h) {
    a <- seq(pi / 2, 3 * pi / 2, length.out = 16)
    r_y <- min(ry, h / 2)
    tibble::tibble(id = id,
                   x = c(x0 + rx + rx * cos(a), x1 - rx + rx * cos(rev(a) + pi)),
                   y = c(yc + r_y * sin(a), yc + r_y * sin(rev(a) + pi)))
  }
  pills <- purrr::pmap_dfr(list(items$i, items$y), \(i, y) pill(i, XM[1], XM[2], y, 1.02)) |>
    left_join(items |> select(id = i, lvl), by = "id")

  ggplot() +
    geom_polygon(data = left, aes(x, y, group = id, fill = paste0("b", lvl)), colour = NA) +
    geom_polygon(data = right, aes(x, y, group = id, fill = paste0("b", lvl)), colour = NA) +
    # the blocks of the UN report and of the full report
    geom_rect(data = un_blocks, aes(xmin = XL[1], xmax = XL[2], ymin = bot, ymax = top),
              fill = ifelse(un_blocks$un == "X", XC, UNC)) +
    geom_rect(data = rep_blocks, aes(xmin = XR[1], xmax = XR[2], ymin = bot, ymax = top), fill = REPC) +
    geom_text(data = un_blocks, aes(x = XL[1] - 0.025, y = (top + bot) / 2, label = UN_PART[as.character(un)]),
              hjust = 1, family = "Inter", size = 2.9, lineheight = 0.95, colour = INK) +
    geom_text(data = rep_blocks, aes(x = XR[2] + 0.025, y = (top + bot) / 2, label = REP_PART[as.character(rep)]),
              hjust = 0, family = "Inter", size = 2.9, lineheight = 0.95, colour = INK) +
    # the questions, as pills
    geom_polygon(data = pills, aes(x, y, group = id, fill = paste0("p", lvl)), colour = NA) +
    scale_fill_manual(values = FILLS, guide = "none") +
    geom_text(data = items, aes(x = mean(XM), y = y, label = name,
                                colour = lvl == "o", fontface = ifelse(lvl == "o", "italic", "plain")),
              family = "Inter", size = 2.65) +
    scale_colour_manual(values = c(`FALSE` = INK, `TRUE` = MUTED), guide = "none") +
    # column heads
    annotate("text", x = XLIM[1] + 0.02, y = 1.9, label = "UN report A/HRC/56/68", hjust = 0, family = "Inter",
             fontface = "bold", size = 3.3, colour = UNC) +
    annotate("text", x = mean(XM), y = 1.9, label = "The questions", family = "Inter", fontface = "bold",
             size = 3.3, colour = INK) +
    annotate("text", x = XLIM[2] - 0.02, y = 1.9, label = "The full report, v1.9", hjust = 1, family = "Inter",
             fontface = "bold", size = 3.3, colour = REPC) +
    # key
    annotate("rect", xmin = c(-0.12, 0.47, 1.06), xmax = c(-0.07, 0.52, 1.11),
             ymin = min(items$y) - 1.35, ymax = min(items$y) - 0.95, fill = FILL, alpha = 0.6) +
    annotate("text", x = c(-0.055, 0.535, 1.125), y = min(items$y) - 1.15, hjust = 0, family = "Inter",
             size = 2.8, colour = INK,
             label = c("measured by this study", "illustrated by this study", "outside this study")) +
    coord_cartesian(xlim = XLIM, ylim = YLIM, expand = FALSE) +
    theme_void()
}

# GDP per capita (World Bank, 2023) against the compliments of the seven models, as in the short version of the
# study; analysis/wealth.py writes the table
wealth <- read.csv(file.path("..", "analysis", "out", "wealth_by_country.csv")) |>
  mutate(lab = unname(country_lbl[country]))
INC_COL <- c("High income" = "#2a78d6", "Upper middle income" = "#eda100", "Lower middle income" = "#e8452c",
             "Low income" = "#a3261a")
wealth_plot <- ggplot(wealth, aes(gdp_pc, compliment)) +
  geom_smooth(method = "lm", formula = y ~ x, colour = MUTED, fill = GRID, linewidth = 0.5) +
  geom_point(aes(colour = income_group), size = 2.2) +
  ggrepel::geom_text_repel(aes(label = lab, colour = income_group), family = "Inter", size = 2.6, seed = 2,
                           box.padding = 0.3, point.padding = 0.15, force = 3, max.overlaps = Inf, max.time = 3,
                           min.segment.length = 0.2, segment.colour = "#cfcdc7", show.legend = FALSE) +
  scale_colour_manual(values = INC_COL, breaks = names(INC_COL),
                      labels = c("high income", "upper middle", "lower middle", "low income")) +
  scale_x_log10(breaks = c(500, 1000, 3000, 10000, 30000, 100000),
                labels = c("$500", "$1k", "$3k", "$10k", "$30k", "$100k")) +
  scale_y_continuous(labels = function(x) paste0(x, " %")) +
  coord_cartesian(ylim = c(0, 41)) +
  labs(x = "GDP per person, 2023 (log scale)", y = "answers complimenting his culture")
