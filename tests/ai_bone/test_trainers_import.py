import ast, pathlib

FILES = [
    "ai_bone/merit/estimate_conflict.py",
    "ai_bone/train/partial_label_trainer.py",
    "ai_bone/train/merit_finetune_trainer.py",
]

def test_parse_server_scripts():
    for f in FILES:
        ast.parse(pathlib.Path(f).read_text(encoding="utf-8"))
