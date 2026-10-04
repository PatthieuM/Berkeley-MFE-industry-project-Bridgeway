"""No-login tests of the opt-in WRDS helper."""

from unittest.mock import Mock, patch

import pytest

from shared.universe import wrds_cloud_v2 as wrds


def credentials(tmp_path, text):
    path = tmp_path / "pgpass"
    path.write_text(text)
    path.chmod(0o600)
    return path


def test_first_matching_entry_and_wildcards(tmp_path):
    path = credentials(tmp_path, "*:*:*:researcher:first\\:value\n*:*:*:researcher:second\n")
    assert wrds._load_credentials(path, "researcher") == ("researcher", "first:value")


def test_ambiguous_username_requires_explicit_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    path = credentials(tmp_path, "*:*:*:one:synthetic\n*:*:*:two:synthetic\n")
    with pytest.raises(RuntimeError, match="username"):
        wrds._load_credentials(path)
    assert wrds._load_credentials(path, "two")[0] == "two"


def test_tunnel_checks_host_keys_has_no_hour_limit_and_closes():
    child = Mock()
    child.isalive.return_value = True
    child.expect.side_effect = [0, 1, 2]
    with patch.object(wrds.pexpect, "spawn", return_value=child) as spawn:
        with patch.object(wrds, "_available_local_port", return_value=43210):
            with wrds._wrds_cloud_tunnel("researcher", "synthetic", duo_option="2") as port:
                assert port == 43210
                child.close.assert_not_called()
    args = spawn.call_args.args[1]
    assert "StrictHostKeyChecking=yes" in args
    assert "sleep 3600" not in " ".join(args)
    assert child.sendline.call_args_list[0].args == ("synthetic",)
    assert child.sendline.call_args_list[1].args == ("2",)
    child.close.assert_called_once_with(force=True)


def test_unknown_host_fails_without_sending_password():
    child = Mock()
    child.isalive.return_value = True
    child.expect.return_value = 3
    with patch.object(wrds.pexpect, "spawn", return_value=child):
        with patch.object(wrds, "_available_local_port", return_value=43210):
            with pytest.raises(RuntimeError, match="host key"):
                with wrds._wrds_cloud_tunnel("researcher", "synthetic"):
                    raise AssertionError("must not yield")
    child.sendline.assert_not_called()
    child.close.assert_called_once_with(force=True)
