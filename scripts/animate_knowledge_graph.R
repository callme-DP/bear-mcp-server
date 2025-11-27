#!/usr/bin/env Rscript

# Reads meta_v2.json, graph.json, and llm_concepts.txt to build a tidygraph
# and renders a gganimate video (graph.mp4) showing nodes/edges appearing
# over time. Time fields are normalized to numeric (epoch seconds) and any
# missing time is filled with the earliest available timestamp.

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(tidygraph)
  library(ggraph)
  library(ggplot2)
  library(gganimate)
})

# ---- configuration -----------------------------------------------------------
meta_path <- "src/note_vectors/meta_v2.json"
graph_path <- "src/note_vectors/graph.json"
concept_path <- "out/llm_concepts.txt"
output_file <- "graph.mp4"

if (!file.exists(meta_path)) stop("meta_v2.json not found at ", meta_path)
if (!file.exists(graph_path)) stop("graph.json not found at ", graph_path)
if (!file.exists(concept_path)) stop("llm_concepts.txt not found at ", concept_path)

if (!requireNamespace("av", quietly = TRUE)) {
  stop("Package 'av' is required by gganimate::av_renderer for mp4 output.")
}

# ---- helpers ----------------------------------------------------------------
parse_time_numeric <- function(x) {
  # Accepts character ISO strings or POSIXct, returns numeric epoch seconds
  if (inherits(x, "POSIXct")) return(as.numeric(x))
  if (is.null(x) || length(x) == 0) return(NA_real_)
  suppressWarnings(as.numeric(as.POSIXct(x, tz = "UTC")))
}

collapse_pattern <- function(values) {
  # Build a regex OR pattern or return NULL if empty
  values <- unique(na.omit(values))
  values <- values[nzchar(values)]
  if (length(values) == 0) return(NULL)
  regex(paste(values, collapse = "|"), ignore_case = TRUE)
}

# ---- load data --------------------------------------------------------------
meta_raw <- fromJSON(meta_path, simplifyDataFrame = TRUE)
graph_raw <- fromJSON(graph_path, simplifyDataFrame = TRUE)
concept_lines <- read_lines(concept_path, skip_empty_rows = TRUE)

# Parse concept tokens after ":" to optionally flag related nodes
concept_splits <- str_split(concept_lines, ":", n = 2, simplify = TRUE)
concept_values <- if (ncol(concept_splits) == 0) character(0) else concept_splits[, 2]
concept_terms <- concept_values |>
  str_split(",", simplify = FALSE) |>
  unlist() |>
  str_trim()
concept_terms <- concept_terms[concept_terms != ""]
concept_pattern <- collapse_pattern(concept_terms)

meta_df <- as_tibble(meta_raw) |>
  transmute(
    id,
    meta_created = creation_date,
    meta_modified = modification_date,
    meta_title = title
  )

nodes <- as_tibble(graph_raw$nodes) |>
  left_join(meta_df, by = "id") |>
  mutate(
    primary_title = coalesce(title, meta_title, name, id),
    time_raw = coalesce(created, meta_created, modified, meta_modified),
    time = parse_time_numeric(time_raw),
    concept_hit = if (is.null(concept_pattern)) FALSE else str_detect(primary_title, concept_pattern)
  )

min_time <- if (all(is.na(nodes$time))) {
  as.numeric(Sys.time())
} else {
  min(nodes$time, na.rm = TRUE)
}

nodes <- nodes |>
  mutate(
    time = ifelse(is.na(time), min_time, time),
    time_posix = as.POSIXct(time, origin = "1970-01-01", tz = "UTC")
  )

edge_lookup <- nodes$time
names(edge_lookup) <- nodes$id
edges <- as_tibble(graph_raw$edges) |>
  drop_na(from, to)
edge_time <- pmax(edge_lookup[edges$from], edge_lookup[edges$to], na.rm = TRUE)
edge_time[is.infinite(edge_time) | is.na(edge_time)] <- min_time
edges <- edges |>
  mutate(
    time = edge_time,
    time_posix = as.POSIXct(time, origin = "1970-01-01", tz = "UTC")
  )

# ---- tidygraph and layout ---------------------------------------------------
graph_tbl <- tbl_graph(nodes = nodes, edges = edges, directed = FALSE, node_key = "id") |>
  activate(nodes) |>
  mutate(degree = centrality_degree(mode = "all"))

set.seed(42)

# ---- plot and animate -------------------------------------------------------
p <- ggraph(graph_tbl, layout = "fr") +
  geom_edge_link(
    aes(color = type),
    alpha = 0.25,
    linewidth = 0.4,
    show.legend = FALSE
  ) +
  geom_node_point(
    aes(size = degree + 1, fill = type, shape = concept_hit),
    color = "gray15",
    alpha = 0.9
  ) +
  scale_fill_brewer(palette = "Set2") +
  scale_shape_manual(values = c(21, 24)) +
  guides(size = "none", shape = "none", fill = "none", color = "none") +
  theme_void() +
  labs(
    title = "知识图谱生长动画",
    subtitle = "{format(as.POSIXct(frame_time, origin = \"1970-01-01\", tz = \"UTC\"), \"%Y-%m-%d\")}",
    caption = "Source: meta_v2.json + graph.json + llm_concepts.txt"
  ) +
  transition_time(time) +
  shadow_mark(past = TRUE, future = FALSE, alpha = 0.15)

dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
anim <- animate(
  p,
  renderer = av_renderer(output_file),
  fps = 12,
  duration = 18,
  width = 1200,
  height = 900,
  res = 120
)

invisible(anim)
