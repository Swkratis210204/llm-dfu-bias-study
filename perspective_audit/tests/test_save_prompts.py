"""save_prompts keeps one markdown file with a section per prompt, whichever experiment writes it."""
from perspective_audit import save_prompts


def test_sections_are_added_replaced_and_kept(tmp_path):
    path = tmp_path / "prompts.md"
    save_prompts(path, {"baseline (6.1)": "ask A", "reworded": "ask B"})
    save_prompts(path, {"scd": "ask C\nmore"})                         # another experiment adds its own section
    save_prompts(path, {"reworded": "ask B2"})                         # a rerun replaces only its own section, in place
    text = path.read_text(encoding="utf-8")
    assert text.index("## baseline (6.1)") < text.index("## reworded") < text.index("## scd")
    assert "ask B2" in text and "ask B\n" not in text and "ask A" in text and "ask C\nmore" in text
    assert text.count("## ") == 3


def test_the_text_above_the_first_section_is_kept(tmp_path):
    path = tmp_path / "prompts.md"
    path.write_text("# Prompts\n\nA note.\n\n## one\n\n```\nold\n```\n", encoding="utf-8")
    save_prompts(path, {"one": "new", "two": "second"})
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Prompts\n\nA note.\n\n## one") and "new" in text and "old" not in text and "## two" in text
