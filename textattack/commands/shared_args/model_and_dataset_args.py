#
# Models hosted on the huggingface model hub.
#
HUGGINGFACE_DATASET_BY_MODEL = {
    #
    # bert-base-uncased
    #
    "bert-base-uncased-ag-news": (
        "textattack/bert-base-uncased-ag-news",
        ("ag_news", None, "test"),
    ),
    "bert-base-uncased-cola": (
        "textattack/bert-base-uncased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "bert-base-uncased-imdb": (
        "textattack/bert-base-uncased-imdb",
        ("imdb", None, "test"),
    ),
    "bert-base-uncased-mnli": (
        "textattack/bert-base-uncased-MNLI",
        ("glue", "mnli", "validation_matched", [1, 2, 0]),
    ),
    "bert-base-uncased-mrpc": (
        "textattack/bert-base-uncased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "bert-base-uncased-qnli": (
        "textattack/bert-base-uncased-QNLI",
        ("glue", "qnli", "validation"),
    ),
    "bert-base-uncased-qqp": (
        "textattack/bert-base-uncased-QQP",
        ("glue", "qqp", "validation"),
    ),
    "bert-base-uncased-rte": (
        "textattack/bert-base-uncased-RTE",
        ("glue", "rte", "validation"),
    ),
    "bert-base-uncased-sst2": (
        "textattack/bert-base-uncased-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "bert-base-uncased-stsb": (
        "textattack/bert-base-uncased-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "bert-base-uncased-wnli": (
        "textattack/bert-base-uncased-WNLI",
        ("glue", "wnli", "validation"),
    ),
    "bert-base-uncased-mr": (
        "textattack/bert-base-uncased-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "bert-base-uncased-snli": (
        "textattack/bert-base-uncased-snli",
        ("snli", None, "test", [1, 2, 0]),
    ),
    "bert-base-uncased-yelp": (
        "textattack/bert-base-uncased-yelp-polarity",
        ("yelp_polarity", None, "test"),
    ),
    #
    # distilbert-base-cased
    #
    "distilbert-base-cased-cola": (
        "textattack/distilbert-base-cased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "distilbert-base-cased-mrpc": (
        "textattack/distilbert-base-cased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "distilbert-base-cased-qqp": (
        "textattack/distilbert-base-cased-QQP",
        ("glue", "qqp", "validation"),
    ),
    "distilbert-base-cased-snli": (
        "textattack/distilbert-base-cased-snli",
        ("snli", None, "test"),
    ),
    "distilbert-base-cased-sst2": (
        "textattack/distilbert-base-cased-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "distilbert-base-cased-stsb": (
        "textattack/distilbert-base-cased-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    #
    # distilbert-base-uncased
    #
    "distilbert-base-uncased-ag-news": (
        "textattack/distilbert-base-uncased-ag-news",
        ("ag_news", None, "test"),
    ),
    "distilbert-base-uncased-cola": (
        "textattack/distilbert-base-cased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "distilbert-base-uncased-imdb": (
        "textattack/distilbert-base-uncased-imdb",
        ("imdb", None, "test"),
    ),
    "distilbert-base-uncased-mnli": (
        "textattack/distilbert-base-uncased-MNLI",
        ("glue", "mnli", "validation_matched", [1, 2, 0]),
    ),
    "distilbert-base-uncased-mr": (
        "textattack/distilbert-base-uncased-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "distilbert-base-uncased-mrpc": (
        "textattack/distilbert-base-uncased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "distilbert-base-uncased-qnli": (
        "textattack/distilbert-base-uncased-QNLI",
        ("glue", "qnli", "validation"),
    ),
    "distilbert-base-uncased-rte": (
        "textattack/distilbert-base-uncased-RTE",
        ("glue", "rte", "validation"),
    ),
    "distilbert-base-uncased-wnli": (
        "textattack/distilbert-base-uncased-WNLI",
        ("glue", "wnli", "validation"),
    ),
    #
    # roberta-base (RoBERTa is cased by default)
    #
    "roberta-base-ag-news": (
        "textattack/roberta-base-ag-news",
        ("ag_news", None, "test"),
    ),
    "roberta-base-cola": (
        "textattack/roberta-base-CoLA",
        ("glue", "cola", "validation"),
    ),
    "roberta-base-imdb": (
        "textattack/roberta-base-imdb",
        ("imdb", None, "test"),
    ),
    "roberta-base-mr": (
        "textattack/roberta-base-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "roberta-base-mrpc": (
        "textattack/roberta-base-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "roberta-base-qnli": (
        "textattack/roberta-base-QNLI",
        ("glue", "qnli", "validation"),
    ),
    "roberta-base-rte": ("textattack/roberta-base-RTE", ("glue", "rte", "validation")),
    "roberta-base-sst2": (
        "textattack/roberta-base-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "roberta-base-stsb": (
        "textattack/roberta-base-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "roberta-base-wnli": (
        "textattack/roberta-base-WNLI",
        ("glue", "wnli", "validation"),
    ),
    #
    # albert-base-v2 (ALBERT is cased by default)
    #
    "albert-base-v2-ag-news": (
        "textattack/albert-base-v2-ag-news",
        ("ag_news", None, "test"),
    ),
    "albert-base-v2-cola": (
        "textattack/albert-base-v2-CoLA",
        ("glue", "cola", "validation"),
    ),
    "albert-base-v2-imdb": (
        "textattack/albert-base-v2-imdb",
        ("imdb", None, "test"),
    ),
    "albert-base-v2-mr": (
        "textattack/albert-base-v2-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "albert-base-v2-rte": (
        "textattack/albert-base-v2-RTE",
        ("glue", "rte", "validation"),
    ),
    "albert-base-v2-qqp": (
        "textattack/albert-base-v2-QQP",
        ("glue", "qqp", "validation"),
    ),
    "albert-base-v2-snli": (
        "textattack/albert-base-v2-snli",
        ("snli", None, "test"),
    ),
    "albert-base-v2-sst2": (
        "textattack/albert-base-v2-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "albert-base-v2-stsb": (
        "textattack/albert-base-v2-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "albert-base-v2-wnli": (
        "textattack/albert-base-v2-WNLI",
        ("glue", "wnli", "validation"),
    ),
    "albert-base-v2-yelp": (
        "textattack/albert-base-v2-yelp-polarity",
        ("yelp_polarity", None, "test"),
    ),
    #
    # xlnet-base-cased
    #
    "xlnet-base-cased-cola": (
        "textattack/xlnet-base-cased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "xlnet-base-cased-imdb": (
        "textattack/xlnet-base-cased-imdb",
        ("imdb", None, "test"),
    ),
    "xlnet-base-cased-mr": (
        "textattack/xlnet-base-cased-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "xlnet-base-cased-mrpc": (
        "textattack/xlnet-base-cased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "xlnet-base-cased-rte": (
        "textattack/xlnet-base-cased-RTE",
        ("glue", "rte", "validation"),
    ),
    "xlnet-base-cased-stsb": (
        "textattack/xlnet-base-cased-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "xlnet-base-cased-wnli": (
        "textattack/xlnet-base-cased-WNLI",
        ("glue", "wnli", "validation"),
    ),
}


#
# Models hosted by textattack.
#
TEXTATTACK_DATASET_BY_MODEL = {
    #
    # LSTMs
    #
    "lstm-ag-news": (
        "models/classification/lstm/ag-news",
        ("ag_news", None, "test"),
    ),
    "lstm-imdb": ("models/classification/lstm/imdb", ("imdb", None, "test")),
    "lstm-mr": (
        "models/classification/lstm/mr",
        ("rotten_tomatoes", None, "test"),
    ),
    "lstm-sst2": ("models/classification/lstm/sst2", ("glue", "sst2", "validation")),
    "lstm-yelp": (
        "models/classification/lstm/yelp",
        ("yelp_polarity", None, "test"),
    ),
    #
    # CNNs
    #
    "cnn-ag-news": (
        "models/classification/cnn/ag-news",
        ("ag_news", None, "test"),
    ),
    "cnn-imdb": ("models/classification/cnn/imdb", ("imdb", None, "test")),
    "cnn-mr": (
        "models/classification/cnn/rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "cnn-sst2": ("models/classification/cnn/sst", ("glue", "sst2", "validation")),
    "cnn-yelp": (
        "models/classification/cnn/yelp",
        ("yelp_polarity", None, "test"),
    ),
    #
    # T5 for translation
    #
    "t5-en-de": (
        "english_to_german",
        ("textattack.datasets.translation.TedMultiTranslationDataset", "en", "de"),
    ),
    "t5-en-fr": (
        "english_to_french",
        ("textattack.datasets.translation.TedMultiTranslationDataset", "en", "fr"),
    ),
    "t5-en-ro": (
        "english_to_romanian",
        ("textattack.datasets.translation.TedMultiTranslationDataset", "en", "de"),
    ),
    #
    # T5 for summarization
    #
    "t5-summarization": ("summarization", ("gigaword", None, "test")),
}
