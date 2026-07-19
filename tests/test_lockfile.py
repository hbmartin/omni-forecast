import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_lock_hosts import unapproved_hosts  # noqa: E402


def test_lockfile_contains_only_approved_package_sources():
    assert unapproved_hosts(Path("uv.lock")) == set()


def test_non_registry_sources_are_rejected(tmp_path):
    """An allowlist that only reads https URLs is blind to every other scheme.

    uv records git, path and workspace dependencies as a source kind rather
    than a URL, so they carry no scheme at all and passed silently.
    """
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        "\n".join(
            (
                "[[package]]",
                'name = "from-git"',
                'source = { git = "ssh://git@github.com/attacker/evil" }',
                "[[package]]",
                'name = "from-path"',
                'source = { path = "../evil" }',
                "[[package]]",
                'name = "insecure"',
                'source = { registry = "http://insecure.example.com/simple" }',
            )
        ),
        encoding="utf-8",
    )
    assert unapproved_hosts(lockfile) == {
        "git source",
        "path source",
        "github.com",
        "insecure.example.com",
    }


def test_the_projects_own_editable_entry_is_allowed(tmp_path):
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        '[[package]]\nname = "self"\nsource = { editable = "." }\n', encoding="utf-8"
    )
    assert unapproved_hosts(lockfile) == set()
