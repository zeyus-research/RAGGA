from ragga.core.dispatch import PropertyWrapper


def test_subscribe_dispatch_unsubscribe():
    received: list[str | None] = []
    prop = PropertyWrapper[str](initial_value="init")
    assert prop.__get__(None, None) == "init"

    prop.subscribe(received.append)
    prop.__set__(None, "a")
    prop.__set__(None, "b")
    assert received == ["a", "b"]
    assert prop.__get__(None, None) == "b"

    prop.unsubscribe(received.append)
    prop.__set__(None, "c")
    assert received == ["a", "b"]
