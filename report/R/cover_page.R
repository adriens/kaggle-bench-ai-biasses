# Full-page A4 cover of the report ("the radial conversation"), drawn from the judged data.
# Sourced by report.qmd (setup chunk), or run from report/: Rscript R/cover_page.R
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
})
if (!exists("rate_table")) source("R/prep.R")

local({
  VERSION <- "v1.7" # the version deposited on Zenodo, shown on the three covers
  BG <- "#0f1115"; FG <- "#f3f1ea"; DIM <- "#8d8b85"; RING <- "#2a2e36"
  CORAL <- "#ff6a4d"; GREEN <- "#2fbf8f"; BLUE <- "#5aa2ff"
  TITLE <- "Space Grotesk"; BODY <- "Inter"

  if (!exists("cover_ring")) source("R/cover_ring.R")
  ring <- cover_ring(if (exists("COVER_CENTRE")) COVER_CENTRE else 1)

  key_item <- function(x, fill, txt) list(
    annotate("rect", xmin = x, xmax = x + 0.035, ymin = 0.35, ymax = 0.65, fill = fill),
    annotate("text", x = x + 0.05, y = 0.5, label = txt, hjust = 0, family = BODY, size = 3.1, colour = FG))
  key <- ggplot() +
    key_item(0.00, CORAL, "outward: \"his origin does not matter\"") +
    key_item(0.38, GREEN, "inward: compliments his culture") +
    annotate("point", x = 0.725, y = 0.5, size = 5, colour = CORAL, alpha = 0.25) +
    annotate("point", x = 0.725, y = 0.5, size = 1.4, colour = "#ffd2c7") +
    annotate("text", x = 0.745, y = 0.5, label = "links him to a risk", hjust = 0, family = BODY,
             size = 3.1, colour = FG) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_void() + theme(plot.background = element_rect(fill = BG, colour = NA))

  byline <- ggplot() +
    annotate("segment", x = 0, xend = 0.06, y = 0.5, yend = 0.5, colour = CORAL, linewidth = 1.2) +
    annotate("text", x = 0.08, y = 0.5, hjust = 0, label = "Adrien Sales", family = TITLE,
             fontface = "bold", size = 6.2, colour = FG) +
    # the version on the right of the author's line; COVER_DATE_TOP adds the month, for a document whose
    # caption leaves it out
    annotate("text", x = 1, y = 0.5, hjust = 1, family = BODY, size = 4.2, colour = DIM,
             label = if (exists("COVER_DATE_TOP") && COVER_DATE_TOP)
               paste0(VERSION, "  ·  ", month.name[as.integer(format(Sys.Date(), "%m"))], " ", format(Sys.Date(), "%Y"))
             else VERSION) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_void() + theme(plot.background = element_rect(fill = BG, colour = NA))

  # logo strip: models tested, judge, and Kaggle for the free GPU quota. Units are inches (coord_fixed),
  # so the logos keep their aspect ratio.
  logo <- function(file, x, h, y = STRIP_H / 2) {
    img <- png::readPNG(file.path("logos", file))
    w <- h * dim(img)[2] / dim(img)[1]
    annotation_raster(img, xmin = x, xmax = x + w, ymin = y - h / 2, ymax = y + h / 2)
  }
  # three rows: the models on the first two, the judge and the GPUs below; one Google logo for the three Gemma
  STRIP_W <- 7.6; ROW <- 0.5; STRIP_H <- 3 * ROW; LH <- 0.34
  Y1 <- STRIP_H - ROW / 2; Y1b <- STRIP_H - 1.5 * ROW; Y2 <- ROW / 2
  tag <- function(x, txt, y) annotate("text", x = x, y = y, label = txt, hjust = 0, family = BODY,
                                      size = 2.5, colour = DIM, fontface = "bold")
  name <- function(x, txt, y) annotate("text", x = x, y = y, label = txt, hjust = 0, family = BODY,
                                       size = 3.1, colour = FG)
  strip <- ggplot() +
    annotate("segment", x = 0, xend = STRIP_W, y = STRIP_H, yend = STRIP_H, colour = RING, linewidth = 0.4) +
    tag(0, "MODELS", Y1) + logo("granite.png", 0.62, LH, Y1) + name(0.99, "Granite 4.2 8B", Y1) +
    logo("qwen.png", 2.3, LH, Y1) + name(2.67, "Qwen3.5-9B", Y1) +
    logo("google.png", 3.9, LH, Y1) + name(4.27, "Gemma 4 E2B \u00b7 E4B \u00b7 26B-A4B", Y1) +
    logo("mistral.png", 0.62, LH, Y1b) + name(0.99, "Ministral 3 8B", Y1b) +
    logo("nvidia.png", 2.3, LH, Y1b) + name(2.67, "Nemotron 3.5 Bolt E4B", Y1b) +
    tag(0, "JUDGE", Y2) + logo("qwen.png", 0.62, LH, Y2) + name(1.02, "Qwen3-32B", Y2) +
    tag(2.35, "GPUs", Y2) + logo("kaggle.png", 2.8, LH * 0.82, Y2) + name(3.61, "free quota, 2\u00d7 T4", Y2) +
    coord_fixed(xlim = c(0, STRIP_W), ylim = c(0, STRIP_H), expand = FALSE, clip = "off") +
    theme_void() + theme(plot.background = element_rect(fill = BG, colour = NA))

  # the spacer leaves room for the labels at the top of the ring, which are drawn outside its panel
  gap <- ggplot() + theme_void() + theme(plot.background = element_rect(fill = BG, colour = NA))
  page <- (byline / gap / ring / key / strip) +
    # COVER_GAP widens the spacer on a smaller page, where the labels of the ring would reach the subtitle
    plot_layout(heights = c(0.55, if (exists("COVER_GAP")) COVER_GAP else 0.7, 10.6, 0.6, 2.3)) +
    plot_annotation(
      title = "She Is Alone With a Man",
      subtitle = if (exists("COVER_SUBTITLE")) COVER_SUBTITLE else
        paste0("The situation should change the advice. His nationality should not.\n",
               "How seven open-weight AI models advise a woman alone with a man, in 11 situations."),
      # COVER_CAPTION replaces the long caption on a shorter page, such as the carousel
      caption = if (exists("COVER_CAPTION")) COVER_CAPTION else paste0(
        "Granite 4.2 8B, Qwen3.5-9B, Gemma 4 E2B, E4B and 26B-A4B, Ministral 3 8B and Nemotron 3.5 Bolt E4B\n",
        format(nrow(R), big.mark = ","), " answers in French · ", format(nrow(D), big.mark = ","),
        " rated by an LLM judge (Qwen3-32B) · 24 nationalities · 11 situations · 3 ages\n",
        "Petals: share of answers, mean of the seven models, sorted from compliments (blue) to suspicion (coral).\n",
        "The empty centre is \"a man\" with no nationality: he gets almost neither.\n",
        "All runs used Kaggle's free GPU quota: this work would not have been possible without it.\n\n",
        month.name[as.integer(format(Sys.Date(), "%m"))], " ", format(Sys.Date(), "%Y"),
        "   ·   x.com/rastadidi  ·  dev.to/adriens  ·  github.com/adriens  ·  linkedin.com/in/adrien-sales  ·  kaggle.com/adriensales"),
      theme = theme(
        plot.background = element_rect(fill = BG, colour = NA),
        plot.margin = margin(46, 34, 30, 34),
        plot.title = element_text(family = TITLE, face = "bold", size = 40, colour = FG, hjust = 0),
        # COVER_SUB_SIZE enlarges the subtitle into a hook, readable on a phone (the carousel)
        plot.subtitle = element_text(family = if (exists("COVER_SUB_SIZE")) TITLE else BODY,
                                     size = if (exists("COVER_SUB_SIZE")) COVER_SUB_SIZE else 13,
                                     face = if (exists("COVER_SUB_SIZE")) "bold" else "plain",
                                     colour = if (exists("COVER_SUB_SIZE")) CORAL else "#c9c6bd", lineheight = 1.15,
                                     margin = margin(10, 0, if (exists("COVER_SUB_SIZE")) 14 else 4, 0)),
        plot.caption = element_text(family = BODY, size = 8, colour = DIM, hjust = 0, lineheight = 1.3)))

  dir.create("covers", showWarnings = FALSE)
  # COVER_FILE lets another document (the short version) draw its own cover with the same design
  out <- if (exists("COVER_FILE")) COVER_FILE else "covers/cover_page"
  # COVER_W and COVER_H (inches) let a document of another size, such as the carousel, reuse the cover
  cw <- if (exists("COVER_W")) COVER_W else 8.27; ch <- if (exists("COVER_H")) COVER_H else 11.69
  ggsave(paste0(out, ".pdf"), page, width = cw, height = ch, device = cairo_pdf, bg = BG)
  ggsave(paste0(out, ".png"), page, width = cw, height = ch, dpi = 150, device = png,
         type = "cairo", bg = BG)
})
