"""Tests for CLI and end-to-end pipeline."""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cli import main as cli_main


def _make_sample(path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("the cat sat on the mat\n")
        f.write("the dog ran in the park\n")
        f.write("cat and dog are friends\n")
        f.write("the sun shines in the sky\n")


def test_encode_and_generate(tmpdir=None):
    tmp = tmpdir or tempfile.mkdtemp()
    data_path = os.path.join(tmp, "data.txt")
    out_dir = os.path.join(tmp, "model")
    _make_sample(data_path)

    # Encode
    args = cli_main.build_parser().parse_args([
        "encode", "--input", data_path, "--output", out_dir,
        "--dim", "64", "--layers", "1"
    ])
    cli_main.cmd_encode(args)
    assert os.path.exists(os.path.join(out_dir, "weights.ncw"))
    assert os.path.exists(os.path.join(out_dir, "index.ncmeta"))
    assert os.path.exists(os.path.join(out_dir, "vocab.json"))

    # Info should not crash
    args = cli_main.build_parser().parse_args(["info", "--weights", out_dir])
    cli_main.cmd_info(args)

    # Generate should produce output
    args = cli_main.build_parser().parse_args([
        "generate", "--weights", out_dir, "--prompt", "the cat", "--tokens", "5"
    ])
    cli_main.cmd_generate(args)

    if tmpdir is None:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_encode_and_generate()
    print("CLI test passed")
