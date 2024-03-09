# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Load required packages
library(igraph)
library(readr)
library(SID)
library(pcalg)
library(graph)

# Load metrics from metric file
metrics_file <- "metrics.csv"
metrics <- read_csv(metrics_file)
sid_lower <- c()
sid_upper <- c()
sid <- c()
# Loop through graph and metric files
for (i in 0:99) {
  print(i)
  # Load the graph from .gml file
  g_hat_file <- paste0("graphs/g_hat", i, ".gml")
  g_hat <- read_graph(g_hat_file, format = "graphml")

  # Convert graph to adjacency matrix
  adj_matrix_hat <- as_adjacency_matrix(g_hat)

  # Load the graph from .gml file
  graph_file <- paste0("../data/graph", i, ".gml")
  graph <- read_graph(graph_file, format = "graphml")

  # Convert graph to adjacency matrix
  adj_matrix_gt <- as_adjacency_matrix(graph)


  # Apply structIntervDist() function to adjacency matrix
  result <- structIntervDist(adj_matrix_gt, adj_matrix_hat)
  sid_lower <- c(sid_lower, result$sidLowerBound)
  sid_upper <- c(sid_upper, result$sidUpperBound)
  sid <- c(sid, result$sid)
}
# Add the result as a new column to the metrics dataframe
metrics$sid_lower <- sid_lower
metrics$sid_upper <- sid_upper
metrics$sid <- sid
# Write the updated metrics dataframe back to the CSV file
write_csv(metrics, metrics_file)
