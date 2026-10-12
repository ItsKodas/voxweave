from tests.test_api import make_client


def test_health(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/health").json() == {"status": "ok"}
