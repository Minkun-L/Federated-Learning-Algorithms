# v6-lda-py

Federated Linear Discriminant Analysis (LDA) for distributed classification without sharing raw data.

This package follows the vantage6 algorithm structure with:
- `partial`: computes per-class local sufficient statistics on each node
- `central`: aggregates node statistics and computes global LDA directions
