# Results

Runs are grouped by gold-set size. **Scores from different groups are not comparable** - they were measured on different questions.

## Gold set: 70 questions

| chunker        | retrieval   | rerank   |   Faithfulness |   Context Precision |   Context Recall |   Factual Correctness |
|:---------------|:------------|:---------|---------------:|--------------------:|-----------------:|----------------------:|
| fixed_512      | hybrid      | False    |          0.880 |               0.577 |            0.830 |                 0.504 |
| fixed_512      | dense       | False    |          0.915 |               0.552 |            0.712 |                 0.434 |
| recursive_1000 | dense       | False    |          0.965 |               0.355 |            0.528 |                 0.426 |

## Gold set: 27 questions

| chunker            | retrieval   | rerank   |   Faithfulness |   Context Precision |   Context Recall |   Factual Correctness |
|:-------------------|:------------|:---------|---------------:|--------------------:|-----------------:|----------------------:|
| structural_article | dense       | False    |          0.942 |               0.519 |            0.703 |                   nan |
| structural_article | hybrid      | False    |          0.960 |               0.485 |            0.665 |                   nan |

## Per-category breakdown

| run_id                          | category   |   Faithfulness |   Context Precision |   Context Recall |   Factual Correctness |
|:--------------------------------|:-----------|---------------:|--------------------:|-----------------:|----------------------:|
| fixed_512__dense__n70           | cross_reg  |          0.879 |               0.162 |            0.431 |                 0.365 |
| fixed_512__dense__n70           | generated  |          0.900 |               0.611 |            0.786 |                 0.427 |
| fixed_512__dense__n70           | lexical    |          1.000 |               0.928 |            0.756 |                 0.470 |
| fixed_512__dense__n70           | lookup     |          0.915 |               0.484 |            0.599 |                 0.475 |
| fixed_512__dense__n70           | multi_hop  |          0.987 |               0.519 |            0.724 |                 0.453 |
| fixed_512__dense__n70           | negative   |          1.000 |               0.000 |            0.250 |                 0.395 |
| fixed_512__hybrid__n70          | cross_reg  |          0.718 |               0.196 |            0.460 |                 0.230 |
| fixed_512__hybrid__n70          | generated  |          0.898 |               0.667 |            0.914 |                 0.512 |
| fixed_512__hybrid__n70          | lexical    |          0.954 |               0.764 |            0.889 |                 0.460 |
| fixed_512__hybrid__n70          | lookup     |          0.850 |               0.434 |            0.683 |                 0.544 |
| fixed_512__hybrid__n70          | multi_hop  |          0.943 |               0.573 |            0.933 |                 0.645 |
| fixed_512__hybrid__n70          | negative   |          0.750 |               0.000 |            0.250 |                 0.290 |
| recursive_1000__dense__n70      | cross_reg  |          1.000 |               0.125 |            0.244 |                 0.228 |
| recursive_1000__dense__n70      | generated  |          0.965 |               0.517 |            0.769 |                 0.505 |
| recursive_1000__dense__n70      | lexical    |          0.961 |               0.167 |            0.273 |                 0.203 |
| recursive_1000__dense__n70      | lookup     |          0.963 |               0.477 |            0.600 |                 0.585 |
| recursive_1000__dense__n70      | multi_hop  |          0.970 |               0.168 |            0.231 |                 0.246 |
| recursive_1000__dense__n70      | negative   |          0.900 |               0.125 |            0.500 |                 0.290 |
| structural_article__dense__n27  | cross_reg  |          0.930 |               0.271 |            0.722 |               nan     |
| structural_article__dense__n27  | lexical    |          1.000 |               0.696 |            0.544 |               nan     |
| structural_article__dense__n27  | lookup     |          0.960 |               0.562 |            0.789 |               nan     |
| structural_article__dense__n27  | multi_hop  |          0.958 |               0.514 |            0.667 |               nan     |
| structural_article__dense__n27  | negative   |          0.750 |               0.500 |            0.500 |               nan     |
| structural_article__hybrid__n27 | cross_reg  |          0.958 |               0.250 |            0.564 |               nan     |
| structural_article__hybrid__n27 | lexical    |          0.800 |               0.633 |            0.333 |               nan     |
| structural_article__hybrid__n27 | lookup     |          0.994 |               0.510 |            0.752 |               nan     |
| structural_article__hybrid__n27 | multi_hop  |          0.991 |               0.514 |            0.698 |               nan     |
| structural_article__hybrid__n27 | negative   |          0.857 |               0.500 |            0.750 |               nan     |

See `docs/EVALUATION.md` for the analysis, caveats and cost breakdown.
