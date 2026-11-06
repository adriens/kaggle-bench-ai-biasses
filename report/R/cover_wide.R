# Landscape covers, same design as the report's: the title and the hook on the left, the radial ring on the right.
#   covers/cover_devto.png   2000 × 840 (DEV.to, 1000 × 420 at 2×)
#   covers/cover_social.png  1280 × 640 (GitHub social preview, Kaggle dataset covers)
# Run from report/: Rscript R/cover_wide.R
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
})
if (!exists("rate_table")) source("R/prep.R")
source("R/cover_ring.R")

local({
  BG <- "#0f1115"; FG <- "#f3f1ea"; DIM <- "#8d8b85"; CORAL <- "#ff6a4d"
  TITLE <- "Space Grotesk"; BODY <- "Inter"
  DOI <- "doi.org/10.5281/zenodo.23991568"

  # w, h in inches; s scales the text to the height of the page; pad (points) shrinks the ring, whose labels
  # reach beyond its panel
  wide <- function(w, h, s, file, dpi, pad = 0, centre = 0.62 * s) {
    text <- ggplot() +
      annotate("text", x = 0, y = 0.80, hjust = 0, vjust = 0.5, label = "She Is Alone\nWith a Man",
               family = TITLE, fontface = "bold", size = 15 * s, colour = FG, lineheight = 0.92) +
      annotate("text", x = 0, y = 0.47, hjust = 0, vjust = 0.5,
               label = "Only his nationality changes.\nThe AI's advice changes too.",
               family = TITLE, fontface = "bold", size = 7.4 * s, colour = CORAL, lineheight = 1.05) +
      annotate("text", x = 0, y = 0.25, hjust = 0, vjust = 0.5,
               label = paste0("7 open-weight AI models · ", round(nrow(R) / 1000), "k answers · 24 nationalities"),
               family = BODY, size = 4.4 * s, colour = FG) +
      annotate("segment", x = 0, xend = 0.05, y = 0.09, yend = 0.09, colour = CORAL, linewidth = 1.1 * s) +
      annotate("text", x = 0.07, y = 0.09, hjust = 0, label = "Adrien Sales", family = TITLE,
               fontface = "bold", size = 5 * s, colour = FG) +
      annotate("text", x = 0.43, y = 0.09, hjust = 0, label = DOI, family = BODY, size = 3.4 * s, colour = DIM) +
      scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
      scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
      coord_cartesian(clip = "off") +
      theme_void() + theme(plot.background = element_rect(fill = BG, colour = NA))
    # wrapped, so that its margin does not shrink the text panel next to it
    ring <- wrap_elements(full = cover_ring(centre = centre, lab_size = 2.15 * s) +
                            theme(plot.margin = margin(pad, 0, pad, 0)), clip = FALSE) +
      theme(plot.background = element_rect(fill = BG, colour = NA))
    page <- (text | ring) + plot_layout(widths = c(1.25, 1)) +
      plot_annotation(theme = theme(plot.background = element_rect(fill = BG, colour = NA),
                                    plot.margin = margin(30 * s, 30 * s, 26 * s, 46 * s)))
    ggsave(file.path("covers", file), page, width = w, height = h, dpi = dpi, device = png, type = "cairo", bg = BG)
  }
  wide(10, 4.2, 1, "cover_devto.png", 200, pad = 20, centre = 0.5)
  wide(12.8, 6.4, 1.3, "cover_social.png", 100)
})
