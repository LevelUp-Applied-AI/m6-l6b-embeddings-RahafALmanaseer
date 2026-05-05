# Embedding Space Analysis: BBC News Articles

## Dimensionality Reduction Choice
I chose **t-SNE** for this visualization because it is optimized for revealing local clusters. As seen in the generated plot, t-SNE successfully projected the 768-dimensional DistilBERT vectors into 2D while preserving the semantic differences between news topics.

## Interpretation of Results
The visualization of BBC News embeddings reveals distinct semantic structure:
- **Clear Separation:** The categories of **Sport**, **Tech**, **Business**, and **Politics** form well-defined, isolated clusters. This indicates that DistilBERT effectively captures the domain-specific language of each topic.
- **Sport Cluster:** Located in the bottom-right quadrant, these articles are tightly grouped, likely due to a very specific vocabulary (e.g., "manager," "match," "coach") found in the sample data.
- **Tech vs. Business:** These clusters are positioned on the left side but remain distinct. The slight proximity reflects the overlapping nature of these topics in modern reporting (e.g., "online gaming" or "card fraudsters").
- **Politics:** This category occupies the top-center region, showing a clear semantic boundary from the other topics.

## Conclusion
The embedding space reveals that DistilBERT understands document context at a high level. Even with a small sample of 20 articles, the model demonstrates its ability to organize complex meaning into a mathematically consistent structure.