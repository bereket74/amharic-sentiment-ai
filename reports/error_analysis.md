# Aggregate error analysis: Transformer

This report deliberately omits tweet text. It summarizes the selected model's held-out errors without redistributing social-media content.

## Per-class recall

| Class | Recall | Test examples |
|---|---:|---:|
| positive | 0.477 | 438 |
| neutral | 0.839 | 224 |
| negative | 0.391 | 1337 |

## Most frequent confusions

- **negative → neutral:** 751 posts
- **positive → neutral:** 184 posts
- **negative → positive:** 63 posts
- **positive → negative:** 45 posts
- **neutral → positive:** 19 posts
- **neutral → negative:** 17 posts

Counts are descriptive, not causal. Social-media sentiment includes sarcasm, mixed-language posts, cultural context, and annotator disagreement; row-level text is not included in this report.
