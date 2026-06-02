# Exhaustive HDBSCAN

Performs HDBSCAN iteratively to provide full data coverage. At each step it forms links between clusters of current step with clusters of previous step. This ends up forming a tree of clusters capturing the full breadth of the context contained in the data.

It makes for a low compute power and reasonably fast procedure for extracting and structuring context from large data.


### Lib is complete, but still writing and finalizing the docs before formally publishing. Git is available for use as is.