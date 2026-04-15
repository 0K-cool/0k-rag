"""
Security Tests: SQL Injection Prevention (VUL-001)

Tests for the SQL sanitization fixes in rag/indexing/indexer.py
Verifies that SQL injection attack patterns are properly escaped.

Security Standard: OWASP A05:2025 - Injection
CWE: CWE-89 (SQL Injection)
Reference: CLAUDE.md lines 277-290
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag.indexing.indexer import _sanitize_sql_value


class TestSQLSanitization:
    """Test the _sanitize_sql_value() function"""

    def test_sanitize_basic_string(self):
        """Test sanitization of normal strings"""
        assert _sanitize_sql_value("test.md") == "test.md"
        assert _sanitize_sql_value("document.pdf") == "document.pdf"
        assert _sanitize_sql_value("my_file_123.txt") == "my_file_123.txt"

    def test_sanitize_single_quote(self):
        """Test escaping of single quotes"""
        assert _sanitize_sql_value("test'file.md") == "test''file.md"
        assert _sanitize_sql_value("it's a test") == "it''s a test"

    def test_sanitize_sql_injection_basic(self):
        """Test basic SQL injection patterns"""
        # Classic OR-based injection
        malicious = "test.md' OR '1'='1"
        expected = "test.md'' OR ''1''=''1"
        assert _sanitize_sql_value(malicious) == expected

        # UNION-based injection
        malicious = "test.md' UNION SELECT * FROM other_table--"
        expected = "test.md'' UNION SELECT * FROM other_table--"
        assert _sanitize_sql_value(malicious) == expected

    def test_sanitize_sql_injection_delete(self):
        """Test SQL injection attempts for DELETE operations"""
        # Attempt to delete all records
        malicious = "test.md' OR '1'='1"
        safe = _sanitize_sql_value(malicious)

        # After escaping, the query should be:
        # WHERE file_path = 'test.md'' OR ''1''=''1'
        # This matches literally, not as logic expression
        assert safe == "test.md'' OR ''1''=''1"

    def test_sanitize_sql_injection_comment(self):
        """Test SQL injection with comment markers"""
        # SQL comment attempts
        malicious = "test.md'--"
        expected = "test.md''--"
        assert _sanitize_sql_value(malicious) == expected

        malicious = "test.md';--"
        expected = "test.md'';--"
        assert _sanitize_sql_value(malicious) == expected

    def test_sanitize_multiple_quotes(self):
        """Test multiple single quotes in string"""
        malicious = "test''file''name.md"
        expected = "test''''file''''name.md"
        assert _sanitize_sql_value(malicious) == expected

    def test_sanitize_project_injection(self):
        """Test SQL injection in project names"""
        # Attempt to match all projects
        malicious = "TestProject' OR '1'='1"
        expected = "TestProject'' OR ''1''=''1"
        assert _sanitize_sql_value(malicious) == expected

    def test_sanitize_path_traversal_combined(self):
        """Test combination of path traversal and SQL injection"""
        malicious = "../../etc/passwd' OR '1'='1"
        expected = "../../etc/passwd'' OR ''1''=''1"
        assert _sanitize_sql_value(malicious) == expected

    def test_sanitize_empty_string(self):
        """Test empty string handling"""
        assert _sanitize_sql_value("") == ""

    def test_sanitize_non_string_raises_error(self):
        """Test that non-string inputs raise TypeError"""
        with pytest.raises(TypeError, match="Expected string"):
            _sanitize_sql_value(123)

        with pytest.raises(TypeError, match="Expected string"):
            _sanitize_sql_value(None)

        with pytest.raises(TypeError, match="Expected string"):
            _sanitize_sql_value(["test.md"])


class TestSQLInjectionAttackVectors:
    """Test real-world SQL injection attack patterns"""

    def test_attack_vector_boolean_based(self):
        """Test boolean-based blind SQL injection"""
        attacks = [
            "' OR '1'='1",
            "' OR 1=1--",
            "' OR 'x'='x",
            "') OR ('1'='1",
        ]

        for attack in attacks:
            safe = _sanitize_sql_value(attack)
            # All single quotes should be escaped
            assert "''" in safe or safe == attack.replace("'", "''")

    def test_attack_vector_union_based(self):
        """Test UNION-based SQL injection"""
        attacks = [
            "' UNION SELECT NULL--",
            "' UNION ALL SELECT *--",
            "' UNION SELECT chunk_id, content FROM knowledge_base--",
        ]

        for attack in attacks:
            safe = _sanitize_sql_value(attack)
            # Single quotes should be escaped
            assert safe.startswith("''")

    def test_attack_vector_time_based(self):
        """Test time-based blind SQL injection"""
        attacks = [
            "'; WAITFOR DELAY '00:00:05'--",
            "'; SELECT SLEEP(5)--",
        ]

        for attack in attacks:
            safe = _sanitize_sql_value(attack)
            # Semicolon and quotes should be escaped
            assert "''" in safe

    def test_attack_vector_stacked_queries(self):
        """Test stacked query injection"""
        attacks = [
            "'; DROP TABLE knowledge_base--",
            "'; DELETE FROM knowledge_base WHERE '1'='1",
        ]

        for attack in attacks:
            safe = _sanitize_sql_value(attack)
            # All single quotes should be escaped
            assert safe.count("''") >= 1


class TestIntegrationWithLanceDB:
    """Integration tests for SQL injection protection in LanceDB operations"""

    def test_where_clause_construction(self):
        """Test that WHERE clauses are constructed safely"""
        # Simulate what happens in the actual code
        malicious_path = "test.md' OR '1'='1"
        safe_path = _sanitize_sql_value(malicious_path)

        # Build WHERE clause as done in indexer.py
        where_clause = f"file_path = '{safe_path}'"

        # Expected: file_path = 'test.md'' OR ''1''=''1'
        # This matches the literal string, not SQL logic
        expected = "file_path = 'test.md'' OR ''1''=''1'"
        assert where_clause == expected

    def test_delete_clause_construction(self):
        """Test that DELETE WHERE clauses are constructed safely"""
        malicious_path = "test.md'; DELETE FROM knowledge_base WHERE '1'='1"
        safe_path = _sanitize_sql_value(malicious_path)

        # Build DELETE WHERE clause
        where_clause = f"file_path = '{safe_path}'"

        # All quotes should be escaped
        assert where_clause.count("''") >= 3  # At least 3 escaped quotes

    def test_project_clause_construction(self):
        """Test project-based WHERE clauses are safe"""
        malicious_project = "TestProject' OR source_project != '"
        safe_project = _sanitize_sql_value(malicious_project)

        # Build WHERE clause
        where_clause = f"source_project = '{safe_project}'"

        # Quotes should be escaped
        assert "''" in where_clause


class TestEdgeCases:
    """Test edge cases and special characters"""

    def test_unicode_characters(self):
        """Test Unicode characters don't break sanitization"""
        unicode_str = "test_文件.md"
        assert _sanitize_sql_value(unicode_str) == unicode_str

    def test_newlines_and_tabs(self):
        """Test newlines and tabs are preserved"""
        test_str = "test\nfile\t.md"
        assert _sanitize_sql_value(test_str) == test_str

    def test_backslashes(self):
        """Test backslashes are preserved (Windows paths)"""
        test_str = "C:\\Users\\test\\file.md"
        assert _sanitize_sql_value(test_str) == test_str

    def test_very_long_string(self):
        """Test sanitization of very long strings"""
        long_str = "a" * 10000 + "'" + "b" * 10000
        expected = "a" * 10000 + "''" + "b" * 10000
        assert _sanitize_sql_value(long_str) == expected

    def test_only_quotes(self):
        """Test string consisting only of quotes"""
        test_str = "'''''"
        expected = "''''''''''"  # Each quote doubled
        assert _sanitize_sql_value(test_str) == expected


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
