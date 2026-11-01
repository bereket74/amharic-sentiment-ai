# Data and model provenance

## Benchmark data

This project uses the **Amharic (`amh`) subset of AfriSenti**, from the official SemEval-2023 Task 12 repository. The upstream paper describes a human-annotated benchmark with native-speaker annotation. The official repository states **CC BY 4.0**; attribution is required. Data are downloaded from the immutable source revision recorded in `src/amharic_sentiment/data.py` and are not committed or re-published by this project.

- Official release and license: [AfriSenti-SemEval repository](https://github.com/afrisenti-semeval/afrisent-semeval-2023)
- Dataset card and split statistics: [AfriSenti on Hugging Face](https://huggingface.co/datasets/shmuhammad/AfriSenti-twitter-sentiment)
- Benchmark paper: [Muhammad et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.862/)

AfriSenti consists of social-media posts. The upstream card says user mentions were replaced with `@user` and URLs removed, while acknowledging incomplete documentation for several curation fields. Labels are positive, neutral, or negative. Dataset predictions can reflect annotator disagreement, social/political context, slang, sarcasm, mixed-language text, and the benchmark's time/domain; they are not ground truth about individuals or groups. Do not use outputs to make high-impact decisions or profile people.

## Transformer initialization

The transformer baseline is initialized from [`Davlan/afro-xlmr-mini`](https://huggingface.co/Davlan/afro-xlmr-mini), whose model card states MIT licensing and adaptation of XLM-R mini on 17 African languages, including Amharic. The exact Hub revision is pinned in `src/amharic_sentiment/models/transformer.py`. Validation selects the training duration on the train subset; the final classifier is then refit on train plus validation. Its upstream paper is [Alabi et al., COLING 2022](https://aclanthology.org/2022.coling-1.382/).

## Code license versus data/model terms

The repository's MIT license applies to original project code only. It does not replace upstream dataset or pretrained-model license terms. The trained classifier is released with explicit source attribution and its own model card; no tweet text or dataset files are included in the model repository or demo package.
