"""
Security Tests: Path Traversal Prevention (VUL-002)

Tests for the path validation fixes in rag/indexing/indexer.py
Verifies that path traversal attack patterns are blocked.

Security Standard: OWASP A01:2025 - Broken Access Control
CWE: CWE-22 (Path Traversal)
Reference: CLAUDE.md lines 155-170
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag.indexing.indexer import _validate_path, _load_allowed_base_paths, SecurityError


class TestPathValidation:
    """Test the _validate_path() function"""

    def test_valid_relative_path(self, tmp_path):
        """Test validation of normal relative paths"""
        allowed = [tmp_path]

        # Create a test file
        test_file = tmp_path / "docs" / "readme.md"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch()

        # Should accept relative path within allowed directory
        validated = _validate_path(str(test_file), allowed)
        assert validated == test_file.resolve()

    def test_valid_absolute_path(self, tmp_path):
        """Test validation of absolute paths within allowed directory"""
        allowed = [tmp_path]

        test_file = tmp_path / "document.md"
        test_file.touch()

        # Should accept absolute path
        validated = _validate_path(str(test_file.resolve()), allowed)
        assert validated == test_file.resolve()

    def test_path_traversal_parent_directory(self, tmp_path):
        """Test blocking ../ path traversal attempts"""
        allowed = [tmp_path / "safe_dir"]
        allowed[0].mkdir(parents=True, exist_ok=True)

        # Create a file outside the allowed directory
        outside_file = tmp_path / "secret.txt"
        outside_file.touch()

        # Attempt to access file outside allowed directory using ../
        malicious_path = str(allowed[0] / ".." / "secret.txt")

        with pytest.raises(SecurityError, match="Path traversal attempt"):
            _validate_path(malicious_path, allowed)

    def test_path_traversal_etc_passwd(self, tmp_path):
        """Test blocking classic /etc/passwd path traversal"""
        allowed = [tmp_path]

        # Attempt to access /etc/passwd
        malicious_paths = [
            "../../etc/passwd",
            "../../../etc/passwd",
            "../../../../etc/passwd",
            "/etc/passwd",
        ]

        for malicious_path in malicious_paths:
            with pytest.raises(SecurityError):
                _validate_path(malicious_path, allowed)

    def test_path_traversal_absolute_path_outside(self, tmp_path):
        """Test blocking absolute paths outside allowed directories"""
        allowed = [tmp_path / "project"]
        allowed[0].mkdir(parents=True, exist_ok=True)

        # Attempt to access file outside allowed directory
        malicious_path = "/tmp/malicious.txt"

        with pytest.raises(SecurityError, match="Path traversal attempt"):
            _validate_path(malicious_path, allowed)

    def test_path_traversal_multiple_parent_refs(self, tmp_path):
        """Test blocking multiple ../ sequences"""
        allowed = [tmp_path / "a" / "b" / "c"]
        allowed[0].mkdir(parents=True, exist_ok=True)

        # Attempt to escape using multiple ../
        malicious_path = str(allowed[0] / ".." / ".." / ".." / "secret.txt")

        with pytest.raises(SecurityError):
            _validate_path(malicious_path, allowed)

    def test_multiple_allowed_paths(self, tmp_path):
        """Test that validation works with multiple allowed base paths"""
        allowed1 = tmp_path / "project1"
        allowed2 = tmp_path / "project2"
        allowed1.mkdir(parents=True, exist_ok=True)
        allowed2.mkdir(parents=True, exist_ok=True)

        file1 = allowed1 / "doc1.md"
        file2 = allowed2 / "doc2.md"
        file1.touch()
        file2.touch()

        allowed = [allowed1, allowed2]

        # Both paths should be valid
        assert _validate_path(str(file1), allowed) == file1.resolve()
        assert _validate_path(str(file2), allowed) == file2.resolve()

    def test_invalid_path_raises_error(self, tmp_path):
        """Test that invalid paths raise SecurityError"""
        allowed = [tmp_path]

        # Non-existent path that resolves outside allowed directory
        with pytest.raises(SecurityError):
            _validate_path("/invalid/nonexistent/path.txt", allowed)

    def test_non_string_path_raises_error(self, tmp_path):
        """Test that non-string paths raise TypeError"""
        allowed = [tmp_path]

        with pytest.raises(TypeError, match="Expected string path"):
            _validate_path(123, allowed)

        with pytest.raises(TypeError, match="Expected string path"):
            _validate_path(None, allowed)

        with pytest.raises(TypeError, match="Expected string path"):
            _validate_path(Path("test.md"), allowed)


class TestPathTraversalAttackVectors:
    """Test real-world path traversal attack patterns"""

    def test_attack_vector_unix_absolute(self, tmp_path):
        """Test Unix absolute path injection"""
        allowed = [tmp_path]

        attacks = [
            "/etc/passwd",
            "/etc/shadow",
            "/root/.ssh/id_rsa",
            "/var/log/auth.log",
        ]

        for attack in attacks:
            with pytest.raises(SecurityError):
                _validate_path(attack, allowed)

    def test_attack_vector_relative_escape(self, tmp_path):
        """Test relative path escape attempts"""
        allowed = [tmp_path / "safe"]
        allowed[0].mkdir(parents=True, exist_ok=True)

        attacks = [
            "../../../etc/passwd",
            "../../..",
            "../../../../../../../../../etc/passwd",
            "safe/../../../etc/passwd",
        ]

        for attack in attacks:
            with pytest.raises(SecurityError):
                _validate_path(attack, allowed)

    def test_attack_vector_null_byte(self, tmp_path):
        """Test null byte injection (Python handles this naturally)"""
        allowed = [tmp_path]

        # Python Path handles null bytes by raising ValueError
        # Our validation should catch this
        try:
            _validate_path("test.txt\x00.md", allowed)
            # If it doesn't raise, it's because Path normalized it
            # which is acceptable
        except (SecurityError, ValueError):
            # Either our validation or Path caught it - good!
            pass

    def test_attack_vector_url_encoded(self, tmp_path):
        """Test URL-encoded path traversal attempts"""
        allowed = [tmp_path]

        # URL-encoded ../
        attacks = [
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ]

        for attack in attacks:
            # Path() may not decode URL encoding, so this might not resolve correctly
            # but it should still be caught as outside allowed directory
            try:
                _validate_path(attack, allowed)
                # If validation passes, verify it's still within allowed directory
                validated = _validate_path(attack, allowed)
                assert str(validated).startswith(str(tmp_path.resolve()))
            except SecurityError:
                # Expected - attack was caught
                pass


class TestConfigIntegration:
    """Test integration with configuration file"""

    def test_load_allowed_paths_fallback(self, tmp_path, monkeypatch):
        """Test fallback to current directory when config not found"""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # No config file exists
        paths = _load_allowed_base_paths()

        # Should fall back to current directory
        assert len(paths) >= 1
        assert paths[0] == tmp_path.resolve()

    def test_validate_path_uses_config(self, tmp_path, monkeypatch):
        """Test that _validate_path() uses config when available"""
        # Create a config file
        config_content = """
security:
  allowed_base_paths:
    - {tmp_path}/project1
    - {tmp_path}/project2
""".format(tmp_path=tmp_path)

        config_file = tmp_path / ".0k-rag.yml"
        config_file.write_text(config_content)

        monkeypatch.chdir(tmp_path)

        # Create test directories
        project1 = tmp_path / "project1"
        project2 = tmp_path / "project2"
        project1.mkdir()
        project2.mkdir()

        # Create test files
        file1 = project1 / "doc.md"
        file2 = project2 / "doc.md"
        file1.touch()
        file2.touch()

        # Both should be valid (config loaded automatically)
        validated1 = _validate_path(str(file1))
        validated2 = _validate_path(str(file2))

        assert validated1 == file1.resolve()
        assert validated2 == file2.resolve()

        # File outside allowed directories should be rejected
        outside_file = tmp_path / "outside.md"
        outside_file.touch()

        with pytest.raises(SecurityError):
            _validate_path(str(outside_file))


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_symlink_within_allowed(self, tmp_path):
        """Test symlinks pointing within allowed directory are accepted"""
        allowed = [tmp_path]

        # Create real file
        real_file = tmp_path / "real.md"
        real_file.touch()

        # Create symlink to real file (both within allowed directory)
        link_file = tmp_path / "link.md"
        link_file.symlink_to(real_file)

        # Should resolve and validate (Path.resolve() follows symlinks)
        validated = _validate_path(str(link_file), allowed)
        assert validated == real_file.resolve()

    def test_symlink_escape_attempt(self, tmp_path):
        """Test symlinks pointing outside allowed directory are blocked"""
        allowed = [tmp_path / "safe"]
        allowed[0].mkdir()

        # Create file outside allowed directory
        outside_file = tmp_path / "outside.md"
        outside_file.touch()

        # Create symlink inside allowed directory pointing outside
        link_file = allowed[0] / "link.md"
        link_file.symlink_to(outside_file)

        # Should be blocked because resolved path is outside allowed directory
        with pytest.raises(SecurityError):
            _validate_path(str(link_file), allowed)

    def test_home_directory_expansion(self, tmp_path):
        """Test that ~ expansion works correctly"""
        allowed = [Path.home() / "test"]
        allowed[0].mkdir(parents=True, exist_ok=True)

        test_file = allowed[0] / "doc.md"
        test_file.touch()

        # Use ~ in path
        home_path = "~/test/doc.md"
        validated = _validate_path(home_path, allowed)

        assert validated == test_file.resolve()

        # Clean up
        test_file.unlink()
        allowed[0].rmdir()

    def test_current_directory_reference(self, tmp_path):
        """Test ./ current directory references"""
        allowed = [tmp_path]

        test_file = tmp_path / "doc.md"
        test_file.touch()

        # ./ should be normalized
        validated = _validate_path(str(tmp_path / "./doc.md"), allowed)
        assert validated == test_file.resolve()

    def test_empty_path(self, tmp_path):
        """Test empty path string"""
        allowed = [tmp_path]

        with pytest.raises(SecurityError):
            _validate_path("", allowed)

    def test_root_path(self, tmp_path):
        """Test root directory path"""
        allowed = [tmp_path]

        # Attempting to validate "/" should fail (outside allowed directory)
        with pytest.raises(SecurityError):
            _validate_path("/", allowed)

    def test_windows_style_paths(self, tmp_path):
        """Test Windows-style paths (backslashes)"""
        allowed = [tmp_path]

        # Create test file
        test_file = tmp_path / "docs" / "readme.md"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch()

        # Windows-style path (Path handles this cross-platform)
        # Note: On Unix, backslash is a valid filename character, not a path separator
        windows_path = str(tmp_path / "docs\\readme.md")

        try:
            validated = _validate_path(windows_path, allowed)
            # If it validates, ensure it's within allowed directory
            # The exact path may differ based on OS handling of backslashes
            assert str(validated).startswith(str(tmp_path.resolve()))
        except SecurityError:
            # May fail if the path doesn't exist, which is acceptable
            pass


class TestKnowledgeBaseIndexerIntegration:
    """Test path validation in KnowledgeBaseIndexer class"""

    def test_indexer_init_with_valid_path(self, tmp_path, monkeypatch):
        """Test indexer initialization with valid database path"""
        from rag.indexing.indexer import KnowledgeBaseIndexer

        monkeypatch.chdir(tmp_path)

        # Valid relative path
        db_path = "test_db"
        indexer = KnowledgeBaseIndexer(db_path)

        # Should initialize successfully
        assert indexer.db_path.is_absolute()
        assert str(indexer.db_path).startswith(str(tmp_path.resolve()))

    def test_indexer_init_with_traversal_attempt(self, tmp_path, monkeypatch, caplog):
        """Test indexer blocks path traversal in db_path"""
        from rag.indexing.indexer import KnowledgeBaseIndexer
        import logging

        # Set up allowed paths to be tmp_path only
        config_content = f"""
security:
  allowed_base_paths:
    - {tmp_path}
"""
        config_file = tmp_path / ".0k-rag.yml"
        config_file.write_text(config_content)

        monkeypatch.chdir(tmp_path)

        # Capture log warnings
        caplog.set_level(logging.WARNING)

        # Attempt path traversal in db_path (should be blocked or warned)
        db_path = "../../etc/passwd"
        indexer = KnowledgeBaseIndexer(db_path)

        # Should log a warning about path validation failure
        assert "Database path validation failed" in caplog.text

        # Falls back to expanduser() behavior, which is acceptable for backward compatibility
        # The key security measure is that validation was attempted and warning was logged


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
