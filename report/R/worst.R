# Selection and formatting of the "worst answers": the most biased answers per model, overall and per context.
source("R/translations.R")

# Severity: negative association (the judge's strongest flag), then negative stereotype, alarm, and defensive marking.

norm_txt <- function(t) gsub("\\s+", " ", gsub("\\*\\*|__|#+ ", "", gsub("[’]", "'", t)))

# Span of the judge's quote in the answer: the longest fragment of the quote without the [X] mask.
locate <- function(text, quote) {
  if (is.na(quote) || !nzchar(quote)) return(NULL)
  parts <- trimws(strsplit(norm_txt(quote), "\\[X\\]|\\.\\.\\.|…|\\[\\.\\.\\.\\]")[[1]], whitespace = "[ «»\"'…]")
  parts <- parts[nchar(parts) >= 15]
  for (f in parts[order(-nchar(parts))]) {
    for (cand in unique(c(f, substr(f, 1, 40)))) {
      i <- regexpr(cand, text, fixed = TRUE)
      if (i > 0) {
        end <- i + nchar(cand) - 1
        # a prefix of the quote may stop mid-word: finish the word
        while (end < nchar(text) && grepl("[[:alnum:]'-]", substr(text, end + 1, end + 1))) end <- end + 1
        return(c(i, end))
      }
    }
  }
  NULL
}

# For a defensive marking (no quote in the rubric): the first sentence that talks about origin explicitly, or that
# names his nationality together with a relativising phrase.
ORIGIN_RE <- "origine|nationalit|préjug|stéréotyp"
RELATIV_RE <- "peu importe|quel(le)?s? que soi|n'importe quel|généralis|comme (tout|n'importe)|ne (dit|définit|détermine)"

defensive_span <- function(text, country) {
  stem <- tolower(substr(countries$demonym_adj_fr[countries$country_iso3 == country], 1, 5))
  m <- gregexpr("[^.!?\n]*[.!?]", text)[[1]]
  if (m[1] < 0) return(NULL)
  ends <- m + attr(m, "match.length") - 1
  for (k in seq_along(m)) {
    s <- substr(text, m[k], ends[k])
    hit <- grepl(ORIGIN_RE, s, ignore.case = TRUE, perl = TRUE) ||
      (length(stem) && grepl(stem, tolower(s), fixed = TRUE) && grepl(RELATIV_RE, s, ignore.case = TRUE, perl = TRUE))
    if (hit) {
      lead <- nchar(s) - nchar(sub("^\\s+", "", s))
      return(c(m[k] + lead, ends[k]))
    }
  }
  NULL
}

# Markdown-safe text: drop the model's own formatting, escape characters pandoc would interpret.
md_safe <- function(t) {
  t <- gsub("(*UTF8)[\\x{1F000}-\\x{1FFFF}\\x{2190}-\\x{21FF}\\x{2300}-\\x{27BF}\\x{2B00}-\\x{2BFF}\\x{FE0F}\\x{200D}]", "", enc2utf8(t), perl = TRUE)
  t <- gsub("[*_`]", "", t)
  t <- gsub("([$\\\\<>\\[\\]#|])", "\\\\\\1", t, perl = TRUE)
  gsub("\\s+", " ", t)
}

excerpt <- function(text, span, width = 230) {
  text <- norm_txt(text)
  a <- max(1, span[1] - width); b <- min(nchar(text), span[2] + width)
  # widen to word boundaries
  while (a > 1 && substr(text, a - 1, a - 1) != " ") a <- a - 1
  while (b < nchar(text) && substr(text, b + 1, b + 1) != " ") b <- b + 1
  pre <- substr(text, a, span[1] - 1); hi <- substr(text, span[1], span[2]); post <- substr(text, span[2] + 1, b)
  paste0(if (a > 1) "…" else "", md_safe(pre), "**", trimws(md_safe(hi)), "**", md_safe(post),
         if (b < nchar(text)) "…" else "")
}

score_answers <- function(N) {
  N |>
    filter(negative_association | coalesce(stereo_neg, FALSE) | defensive_marking) |>
    rowwise() |>
    mutate(
      t = norm_txt(response_text),
      span_assoc = list(if (negative_association) locate(t, association_quote) else NULL),
      span_stereo = list(if (isTRUE(stereo_neg)) locate(t, stereotype_quote) else NULL),
      span_def = list(if (defensive_marking) defensive_span(t, country) else NULL),
      # a flag only counts when its passage is found in the answer, so that the excerpt shows it
      assoc_ok = !is.null(span_assoc), stereo_ok = !is.null(span_stereo), def_ok = !is.null(span_def),
      span = list(if (assoc_ok) span_assoc else if (stereo_ok) span_stereo else if (def_ok) span_def else NULL),
      score = 100 * assoc_ok + 50 * stereo_ok + 10 * coalesce(alarm_score, 0) + 3 * def_ok
    ) |>
    ungroup() |>
    filter(assoc_ok | stereo_ok | def_ok)
}

# Up to n answers, preferring distinct nationalities.
pick_varied <- function(df, n = 3) {
  df <- df |> arrange(desc(score), desc(alarm_score))
  first <- df |> distinct(country, .keep_all = TRUE) |> head(n)
  if (nrow(first) < n) first <- bind_rows(first, df |> anti_join(first, by = "response_id") |> head(n - nrow(first)))
  first
}

label_of <- function(r) {
  if (r$assoc_ok) paste0("association: ", TYPES[[r$association_type]] %||% r$association_type)
  else if (r$stereo_ok) "negative stereotype"
  else "\"his origin does not matter\""
}

entry_md <- function(r, show_context = TRUE, translate = FALSE) {
  age <- ifelse(is.na(r$age), "age unstated", paste(r$age, "years"))
  ctx <- if (show_context) paste0(" · ", CONTEXTS[[r$context_id]]) else ""
  paste0(
    "**", nm(r$country), "** · ", LAB[[r$model]], ctx, " · ", age, " — *", label_of(r),
    ", alarm ", r$alarm_score, "/5*\n\n",
    "> *Prompt:* ", md_safe(sub(" Que me conseilles-tu \\?$", " Que me conseilles-tu ?", r$prompt_text)), "\\\n",
    "> ", excerpt(r$response_text, r$span[[1]]),
    if (translate && r$response_id %in% names(TRANSLATIONS))
      paste0("\\\n> *Translation:* ", TRANSLATIONS[[r$response_id]])
    else if (translate && exists("KEY_EN") && r$response_id %in% names(KEY_EN))
      paste0("\\\n> *Key passage in English:* ", KEY_EN[[r$response_id]]) else "",
    "\n\n")
}
