import spacy

from spacylize import analysis

nlp = spacy.blank("en")


def test_find_duplicates_exact_and_near():
    texts = [
        "The red wireless headphones offer great sound and long battery life.",
        "the red  wireless headphones offer great sound and long battery life.",
        "The red wireless headphones offer great sound and a long battery life.",
        "A wooden dining table with four matching chairs.",
    ]
    X = analysis.embed(texts)

    duplicates = analysis.find_duplicates(texts, X)

    assert duplicates["exact"] == [[0, 1]]
    assert {(a, b) for a, b, _ in duplicates["near"]} == {(0, 2), (1, 2)}


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


def test_project_2d_handles_small_and_larger_datasets():
    for n in (2, 5, 30):
        texts = [f"document number {i} about topic {i % 3}" for i in range(n)]
        coords = analysis.project_2d(analysis.embed(texts))
        assert coords.shape == (n, 2)


def test_find_off_topic_reads_scores_from_generate():
    docs = [nlp("A new laptop."), nlp("Bake the bread."), nlp("No score.")]
    docs[0].user_data["topic_score"] = 0.99
    docs[1].user_data["topic_score"] = 0.01

    result = analysis.find_off_topic(docs)

    assert result == {
        "checked": 2,
        "off_topic": [{"index": 1, "score": 0.01, "text": "Bake the bread."}],
    }
    assert analysis.find_off_topic([nlp("No score.")]) is None
