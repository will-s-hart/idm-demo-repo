# Note that AI tools were used to generate tests
import idmdemo


def test_public_api():
    assert set(idmdemo.__all__) == set()
