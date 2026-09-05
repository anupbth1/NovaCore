"""Tests for chat session."""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from novacore.inference.chat import ChatSession


def _build_sample_model(tmpdir):
    from cli import main as cli_main
    data_path = os.path.join(tmpdir, "data.txt")
    out_dir = os.path.join(tmpdir, "model")
    with open(data_path, 'w', encoding='utf-8') as f:
        f.write("the cat sat on the mat and watched the world\n")
        f.write("the dog ran through the park chasing a ball\n")
        f.write("machine learning helps computers learn from data\n")
        f.write("the cat purred while on the warm mat\n")
        f.write("the world is a beautiful place\n")
    args = cli_main.build_parser().parse_args([
        "encode", "--input", data_path, "--output", out_dir,
        "--dim", "64", "--layers", "1"
    ])
    cli_main.cmd_encode(args)
    return out_dir


def test_chat_session(tmpdir=None):
    tmp = tmpdir or tempfile.mkdtemp()
    model_dir = _build_sample_model(tmp)
    session = ChatSession(model_dir)
    session.load()
    assert session._loaded is True

    reply = session.chat("what is a cat", max_tokens=10)
    assert isinstance(reply, str) and len(reply) > 0
    # Reply should not echo the full prompt
    assert "what is a cat" not in reply.lower()

    assert len(session.get_history()) == 2  # user + assistant
    session.reset()
    assert len(session.get_history()) == 0

    if tmpdir is None:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_chat_session()
    print("chat test PASSED")
