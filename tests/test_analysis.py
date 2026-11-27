import spacy

from spacylize import analysis

nlp = spacy.blank("en")


def test_duplicate_ids_ignores_case_punctuation_and_spacing():
    texts = [
        "The red headphones offer great sound.",
        "the red  headphones offer great sound",
        "The red headphones offer a great sound.",
        "The red headphones offer great sound.",
    ]

    # The first text is kept; a different wording is not a duplicate
    assert analysis.duplicate_ids(texts) == {1, 3}


def test_diversity_distinct_n():
    docs = [nlp("a b a"), nlp("a b c")]

    result = analysis.diversity(docs)

    # unigrams: a b a a b c -> 3 unique of 6; bigrams: ab ba ab bc -> 3 unique of 4
    assert result["distinct_1"] == 0.5
    assert result["distinct_2"] == 0.75
    assert result["top_openings"] == [("a b a", 1), ("a b c", 1)]


def test_label_quality_ner_conflicts():
    doc_a = nlp("Apple makes phones")
    doc_a.ents = [doc_a.char_span(0, 5, label="BRAND")]
    doc_b = nlp("Apple is red")
    doc_b.ents = [doc_b.char_span(0, 5, label="COLOR")]

    result = analysis.label_quality([doc_a, doc_b], "ner")

    assert result["by_label"]["BRAND"]["unique"] == 1
    assert result["conflicts"] == [{"text": "apple", "labels": ["BRAND", "COLOR"]}]


def test_label_quality_textcat():
    docs = [nlp("one"), nlp("two"), nlp("three")]
    docs[0].cats = {"A": 1.0, "B": 0.0}
    docs[1].cats = {"A": 0.0, "B": 0.0}
    docs[2].cats = {"A": 1.0, "B": 1.0}

    result = analysis.label_quality(docs, "textcat")

    assert result == {"no_category": 1, "multiple_categories": 1}


def test_check_labels():
    doc = nlp("Apple makes phones")
    doc.ents = [doc.char_span(0, 5, label="BRAND")]

    result = analysis.check_labels([doc], "ner", ["BRAND", "COLOR"])

    assert result == {
        "counts": {"BRAND": 1, "COLOR": 0},
        "missing": ["COLOR"],
        "unknown": [],
    }


def test_off_topic_ids_reads_scores_from_generate():
    docs = [nlp("A new laptop."), nlp("Bake the bread."), nlp("No score.")]
    docs[0].user_data["topic_score"] = 0.99
    docs[1].user_data["topic_score"] = 0.01

    assert analysis.off_topic_ids(docs) == {1}


def test_find_issues_runs_each_check():
    docs = [
        nlp("Apple makes phones"),
        nlp("Apple makes phones"),  # duplicate
        nlp("Apple is red"),  # label conflict with "Apple"
        nlp("Bake the bread"),  # off-topic
        nlp("A [laptop] here"),  # leftover markup
    ]
    docs[0].ents = [docs[0].char_span(0, 5, label="BRAND")]
    docs[1].ents = [docs[1].char_span(0, 5, label="BRAND")]
    docs[2].ents = [docs[2].char_span(0, 5, label="COLOR")]
    docs[3].user_data["topic_score"] = 0.01

    issues = analysis.find_issues(docs, "ner", labels=["BRAND"])

    assert issues == {
        0: ["label conflict"],
        1: ["duplicate", "label conflict"],
        2: ["label conflict", "unknown label"],
        3: ["off-topic"],
        4: ["leftover markup"],
    }
