"""Live tests for jev.py. Run with `python test_jev.py` or `pytest test_jev.py`."""

from jev import choice, noul, score

CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was completed in 1889 and is 330 metres tall."
)


def test_noul():
    yes = noul(CONTEXT, "The text states when the tower was completed")
    no = noul(CONTEXT, "The text states who designed the tower's elevators")
    print(f"noul  -> completed date: {yes:.2f} | elevator designer: {no:.2f}")
    assert yes > 0.5 > no


def test_choice():
    result = choice(
        CONTEXT,
        "Which question does this passage answer best",
        {
            "height": "How tall is the Eiffel Tower?",
            "tickets": "How much does a ticket to the Eiffel Tower cost?",
            "berlin": "What is the population of Berlin?",
        },
    )
    print(f"choice -> {result['choice']} {result['probabilities']} (conf {result['confidence']})")
    assert result["choice"] == "height"


def test_score():
    result = score(
        CONTEXT,
        "How relevant is this passage to the query 'When was the Eiffel Tower built?'",
        ["Irrelevant", "Partially relevant", "Directly answers the query"],
    )
    print(f"score -> {result['score']} {result['probabilities']} (conf {result['confidence']})")
    # score is the probability-weighted mean, e.g. 1.98, not an integer
    assert round(result["score"]) == 2


if __name__ == "__main__":
    for test in (test_noul, test_choice, test_score):
        try:
            test()
            print(f"  PASS {test.__name__}")
        except AssertionError:
            print(f"  FAIL {test.__name__}")
