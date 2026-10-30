# Cover proposals (16:9 PNG, 2400 x 1350) built from the judged data. Run from report/:
#   Rscript R/covers.R
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggrepel)
})
source("R/prep.R")
OUT <- "covers"
dir.create(OUT, showWarnings = FALSE)
nm <- function(iso) unname(country_lbl[iso])
TITLE <- "Space Grotesk"; BODY <- "Inter"

save_cover <- function(p, name) {
  ggsave(file.path(OUT, paste0(name, ".png")), p, width = 12, height = 6.75, dpi = 200,
         device = png, type = "cairo", bg = p$theme$plot.background$fill %||% "white")
}

per_country <- function(metric) rate_table(metric) |> group_by(country) |>
  summarise(pct = mean(pct), k = sum(k), .groups = "drop")
def <- per_country("defensive_marking")
pos <- per_country("stereo_pos")
neg <- per_country("negative_association")
main_type <- D |> filter(negative_association) |> count(country, association_type) |>
  group_by(country) |> slice_max(n, n = 1, with_ties = FALSE) |> ungroup()

# ---------------------------------------------------------------- A. waffle, light
cat_of <- function(d) case_when(
  d$negative_association ~ "links him to a risk",
  d$defensive_marking ~ "\"his origin does not matter\"",
  d$stereo_pos ~ "compliments his culture",
  d$encounter ~ "\"get to know him\"",
  TRUE ~ "other practical advice")
LEV <- c("links him to a risk", "\"his origin does not matter\"", "compliments his culture",
         "\"get to know him\"", "other practical advice")
PAL <- c("#d63c2a", "#f08a4b", "#1baf7a", "#2a78d6", "#dcd9d1")
to100 <- function(n) { x <- 100 * n / sum(n); f <- floor(x); r <- 100 - sum(f)
  f[order(x - f, decreasing = TRUE)[seq_len(r)]] <- f[order(x - f, decreasing = TRUE)[seq_len(r)]] + 1; f }
PANELS <- c(BASE = "\"I am alone with a man\"", FIN = "\"… with a Finnish man\"",
            NGA = "\"… with a Nigerian man\"")
waf <- D |> filter(country %in% names(PANELS)) |>
  mutate(cat = factor(cat_of(cur_data()), levels = LEV)) |>
  count(country, cat, .drop = FALSE) |>
  group_by(country) |> mutate(sq = to100(n)) |> ungroup() |>
  group_by(country) |> reframe(cat = rep(cat, sq)) |>
  group_by(country) |> mutate(i = row_number() - 1, x = i %/% 10, y = 9 - i %% 10) |> ungroup() |>
  mutate(panel = factor(PANELS[country], levels = PANELS))
pA <- ggplot(waf, aes(x, y, fill = cat)) +
  geom_tile(width = 0.84, height = 0.84) +
  facet_wrap(~ panel, nrow = 1) +
  coord_equal() +
  scale_fill_manual(values = setNames(PAL, LEV), drop = FALSE) +
  labs(title = "Same question. Not the same answer.",
       subtitle = "What two open AI models advise a woman alone with a man, depending only on his nationality. One square = 1 % of answers.",
       caption = "Granite 4.2 8B and Qwen3.5-9B · ~200 judged answers per nationality, ~400 for \"a man\" · 11 situations, 3 ages · LLM judge Qwen3-32B") +
  guides(fill = guide_legend(nrow = 1, override.aes = list(size = 0))) +
  theme_void(base_family = BODY) +
  theme(plot.background = element_rect(fill = "#fbfaf7", colour = NA),
        plot.margin = margin(36, 48, 28, 48),
        plot.title = element_text(family = TITLE, face = "bold", size = 30, colour = "#141414"),
        plot.subtitle = element_text(size = 12.5, colour = "#55534e", margin = margin(8, 0, 22, 0)),
        plot.caption = element_text(size = 8.5, colour = "#8a877f", hjust = 0, margin = margin(18, 0, 0, 0)),
        plot.title.position = "plot", plot.caption.position = "plot",
        strip.text = element_text(family = TITLE, size = 14, face = "bold", colour = "#141414",
                                  margin = margin(0, 0, 10, 0)),
        panel.spacing.x = unit(2.2, "lines"),
        legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 10.5, colour = "#33312d", margin = margin(0, 14, 0, 4)),
        legend.key.size = unit(13, "pt"), legend.margin = margin(16, 0, 0, 0))
save_cover(pA, "cover_A_waffle")

# ---------------------------------------------------------------- B. butterfly, dark
BG <- "#111317"; FG <- "#f2f0ea"; DIM <- "#8b8a86"
bt <- def |> select(country, def = pct) |> inner_join(pos |> select(country, pos = pct), by = "country") |>
  left_join(main_type |> select(country, type = association_type), by = "country") |>
  left_join(neg |> select(country, negk = k), by = "country") |>
  mutate(lab = ifelse(country == "BASE", "a man (no nationality)", nm(country)),
         ord = ifelse(country == "BASE", -Inf, def - pos)) |>
  arrange(ord) |> mutate(lab = factor(lab, levels = lab))
flag <- bt |> filter(country %in% c("NGA", "AFG")) |>
  mutate(note = paste0("+ ", ifelse(type == "scam", "scam", "migration / gender norms"), " warnings"))
pB <- ggplot(bt, aes(y = lab)) +
  geom_col(aes(x = -pos), fill = "#2fbf8f", width = 0.68) +
  geom_col(aes(x = def), fill = "#ff6a4d", width = 0.68) +
  geom_label(aes(x = 0, label = lab), family = BODY, size = 3, colour = FG, fill = BG,
             label.size = 0, label.padding = unit(1.5, "pt")) +
  geom_text(data = flag, aes(x = def + 1.5, label = note), hjust = 0, family = BODY, size = 3,
            colour = "#ff6a4d", fontface = "bold") +
  annotate("text", x = -3, y = nrow(bt) + 1.3, hjust = 1, family = TITLE, size = 4.6, colour = "#2fbf8f",
           fontface = "bold", label = "◀ flattered: compliments his culture") +
  annotate("text", x = 3, y = nrow(bt) + 1.3, hjust = 0, family = TITLE, size = 4.6, colour = "#ff6a4d",
           fontface = "bold", label = "\"his origin does not matter\" ▶") +
  scale_x_continuous(labels = \(x) paste0(abs(x), " %"), limits = c(-62, 100), breaks = seq(-60, 80, 20)) +
  scale_y_discrete(expand = expansion(add = c(0.7, 2.2))) +
  labs(title = "Alone with a stranger",
       subtitle = "Ask an AI for safety advice about a man, and his nationality decides whether it flatters him or defends him.",
       caption = "Share of answers, mean of Granite 4.2 8B and Qwen3.5-9B · ~200 judged answers per nationality · LLM judge Qwen3-32B",
       x = NULL, y = NULL) +
  theme_minimal(base_family = BODY) +
  theme(plot.background = element_rect(fill = BG, colour = NA),
        panel.grid = element_blank(),
        panel.grid.major.x = element_line(colour = "#24272d", linewidth = 0.3),
        axis.text.y = element_blank(), axis.text.x = element_text(colour = DIM, size = 8),
        plot.margin = margin(30, 40, 20, 40),
        plot.title = element_text(family = TITLE, face = "bold", size = 28, colour = FG),
        plot.subtitle = element_text(size = 12, colour = "#c9c6bd", margin = margin(6, 0, 6, 0)),
        plot.caption = element_text(size = 8.5, colour = DIM, hjust = 0),
        plot.title.position = "plot", plot.caption.position = "plot")
save_cover(pB, "cover_B_butterfly")

# ---------------------------------------------------------------- C. spectrum, dark
sp <- bt |> filter(country != "BASE") |> mutate(idx = def - pos) |> arrange(idx) |>
  mutate(side = rep(c(1, -1), length.out = n()), lab = as.character(lab))
base_idx <- bt$def[bt$country == "BASE"] - bt$pos[bt$country == "BASE"]
pC <- ggplot(sp, aes(idx, 0)) +
  annotate("segment", x = -45, xend = 70, y = 0, yend = 0, colour = "#2a2d33", linewidth = 6, lineend = "round") +
  geom_point(aes(colour = idx, size = negk + 2)) +
  geom_text_repel(aes(label = lab), nudge_y = sp$side * 0.55, direction = "x", family = BODY,
                  size = 3.3, colour = FG, segment.colour = "#4a4d55", segment.size = 0.3,
                  min.segment.length = 0, box.padding = 0.25, force = 2, seed = 3, max.overlaps = 50) +
  annotate("point", x = base_idx, y = 0, shape = 8, size = 3.5, colour = FG, stroke = 0.9) +
  annotate("text", x = base_idx, y = -0.17, vjust = 1, family = BODY, size = 3, colour = DIM,
           label = "\"a man\" with no nationality") +
  annotate("text", x = -45, y = 1.35, hjust = 0, family = TITLE, fontface = "bold", size = 6,
           colour = "#5aa2ff", label = "CULTURAL GUIDE") +
  annotate("text", x = -45, y = 1.12, hjust = 0, family = BODY, size = 3.3, colour = DIM,
           label = "ice-breakers, compliments on his culture") +
  annotate("text", x = 70, y = 1.35, hjust = 1, family = TITLE, fontface = "bold", size = 6,
           colour = "#ff6a4d", label = "SECURITY GUARD") +
  annotate("text", x = 70, y = 1.12, hjust = 1, family = BODY, size = 3.3, colour = DIM,
           label = "\"his origin does not matter\", warnings about scams or migration") +
  scale_colour_gradient2(low = "#5aa2ff", mid = "#b9b7b1", high = "#ff6a4d", midpoint = 15, guide = "none") +
  scale_size_area(max_size = 14, guide = "none") +
  scale_y_continuous(limits = c(-1.5, 1.5)) +
  labs(title = "Same question, two AI personas",
       subtitle = "Where each nationality lands when a woman asks an AI for advice about being alone with a man.",
       caption = "Position: share of \"his origin does not matter\" minus share of compliments · bubble size: answers linking him to a risk · Granite 4.2 8B + Qwen3.5-9B, LLM judge Qwen3-32B") +
  theme_void(base_family = BODY) +
  theme(plot.background = element_rect(fill = BG, colour = NA),
        plot.margin = margin(32, 40, 22, 40),
        plot.title = element_text(family = TITLE, face = "bold", size = 28, colour = FG),
        plot.subtitle = element_text(size = 12, colour = "#c9c6bd", margin = margin(6, 0, 0, 0)),
        plot.caption = element_text(size = 8.5, colour = DIM, hjust = 0),
        plot.title.position = "plot", plot.caption.position = "plot")
save_cover(pC, "cover_C_spectrum")
cat("written:", list.files(OUT, full.names = TRUE), sep = "\n")
