from minisgl.server.args import ServerArgs
from minisgl.server.api_server import FrontendManager


def test_apply_prefix_to_plain_prompt():
    manager = FrontendManager(
        config=ServerArgs(
            model_path="dummy",
            dtype=None,  # type: ignore[arg-type]
            tp_info=None,  # type: ignore[arg-type]
            prefix_prompt="prefix",
        ),
        send_tokenizer=None,  # type: ignore[arg-type]
        recv_tokenizer=None,  # type: ignore[arg-type]
    )

    assert manager.apply_prefix("hello") == "prefix\nhello"


def test_apply_prefix_to_chat_messages():
    manager = FrontendManager(
        config=ServerArgs(
            model_path="dummy",
            dtype=None,  # type: ignore[arg-type]
            tp_info=None,  # type: ignore[arg-type]
            prefix_prompt="prefix",
        ),
        send_tokenizer=None,  # type: ignore[arg-type]
        recv_tokenizer=None,  # type: ignore[arg-type]
    )

    assert manager.apply_prefix([{"role": "user", "content": "hello"}]) == [
        {"role": "system", "content": "prefix"},
        {"role": "user", "content": "hello"},
    ]
    assert manager.apply_prefix([{"role": "system", "content": "sys"}]) == [
        {"role": "system", "content": "prefix\nsys"},
    ]
