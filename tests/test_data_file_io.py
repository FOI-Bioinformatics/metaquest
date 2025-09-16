"""
Test data.file_io module functionality.

Tests for file I/O utilities including directory management, file operations, and data processing.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pandas as pd
import pytest

from metaquest.data.file_io import (
    ensure_directory,
    list_files,
    copy_file,
    read_csv,
    write_csv,
    read_json,
    write_json,
    process_files_in_directory,
)
from metaquest.core.exceptions import DataAccessError


class TestEnsureDirectory:
    """Test ensure_directory function."""

    def test_ensure_directory_success(self, tmp_path):
        """Test successful directory creation."""
        new_dir = tmp_path / "new_directory"
        
        result = ensure_directory(new_dir)
        
        assert result == new_dir
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_directory_existing(self, tmp_path):
        """Test with existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        
        result = ensure_directory(existing_dir)
        
        assert result == existing_dir
        assert existing_dir.exists()

    def test_ensure_directory_nested(self, tmp_path):
        """Test creating nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        
        result = ensure_directory(nested_dir)
        
        assert result == nested_dir
        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_ensure_directory_string_path(self, tmp_path):
        """Test with string path."""
        new_dir = str(tmp_path / "string_path")
        
        result = ensure_directory(new_dir)
        
        assert result == Path(new_dir)
        assert Path(new_dir).exists()

    def test_ensure_directory_failure(self):
        """Test failure in directory creation."""
        with patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied")):
            with pytest.raises(DataAccessError, match="Failed to create directory"):
                ensure_directory("/invalid/path")


class TestListFiles:
    """Test list_files function."""

    def test_list_files_basic(self, tmp_path):
        """Test basic file listing."""
        # Create test files
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "file3.csv").touch()
        
        result = list_files(tmp_path, "*.txt")
        
        assert len(result) == 2
        file_names = [f.name for f in result]
        assert "file1.txt" in file_names
        assert "file2.txt" in file_names
        assert "file3.csv" not in file_names

    def test_list_files_all_files(self, tmp_path):
        """Test listing all files with default pattern."""
        # Create test files
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.csv").touch()
        (tmp_path / "file3.json").touch()
        
        result = list_files(tmp_path)
        
        assert len(result) == 3

    def test_list_files_no_matches(self, tmp_path):
        """Test with no matching files."""
        (tmp_path / "file1.txt").touch()
        
        result = list_files(tmp_path, "*.csv")
        
        assert result == []

    def test_list_files_empty_directory(self, tmp_path):
        """Test with empty directory."""
        result = list_files(tmp_path)
        
        assert result == []

    def test_list_files_nonexistent_directory(self):
        """Test with nonexistent directory."""
        result = list_files("/nonexistent/path")
        
        assert result == []

    def test_list_files_string_path(self, tmp_path):
        """Test with string path."""
        (tmp_path / "test.txt").touch()
        
        result = list_files(str(tmp_path), "*.txt")
        
        assert len(result) == 1
        assert result[0].name == "test.txt"


class TestCopyFile:
    """Test copy_file function."""

    def test_copy_file_success(self, tmp_path):
        """Test successful file copy."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        
        source.write_text("test content")
        
        result = copy_file(source, dest)
        
        assert result == dest
        assert dest.exists()
        assert dest.read_text() == "test content"

    def test_copy_file_create_dest_directory(self, tmp_path):
        """Test copy with destination directory creation."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "new_dir" / "dest.txt"
        
        source.write_text("test content")
        
        result = copy_file(source, dest)
        
        assert result == dest
        assert dest.exists()
        assert dest.parent.exists()
        assert dest.read_text() == "test content"

    def test_copy_file_string_paths(self, tmp_path):
        """Test copy with string paths."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        
        source.write_text("test content")
        
        result = copy_file(str(source), str(dest))
        
        assert result == dest
        assert dest.exists()

    def test_copy_file_nonexistent_source(self, tmp_path):
        """Test copy with nonexistent source file."""
        source = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"
        
        with pytest.raises(DataAccessError, match="Failed to copy"):
            copy_file(source, dest)

    def test_copy_file_permission_error(self, tmp_path):
        """Test copy with permission error."""
        source = tmp_path / "source.txt"
        dest = tmp_path / "dest.txt"
        
        source.write_text("test content")
        
        with patch('shutil.copy2', side_effect=OSError("Permission denied")):
            with pytest.raises(DataAccessError, match="Failed to copy"):
                copy_file(source, dest)


class TestReadCsv:
    """Test read_csv function."""

    def test_read_csv_success(self, tmp_path):
        """Test successful CSV reading."""
        csv_file = tmp_path / "test.csv"
        csv_content = "name,age,city\nAlice,25,NYC\nBob,30,LA"
        csv_file.write_text(csv_content)
        
        result = read_csv(csv_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["name", "age", "city"]
        assert result.iloc[0]["name"] == "Alice"

    def test_read_csv_with_kwargs(self, tmp_path):
        """Test CSV reading with additional arguments."""
        csv_file = tmp_path / "test.csv"
        csv_content = "name;age;city\nAlice;25;NYC\nBob;30;LA"
        csv_file.write_text(csv_content)
        
        result = read_csv(csv_file, sep=';')
        
        assert len(result) == 2
        assert list(result.columns) == ["name", "age", "city"]

    def test_read_csv_string_path(self, tmp_path):
        """Test CSV reading with string path."""
        csv_file = tmp_path / "test.csv"
        csv_content = "name,age\nAlice,25"
        csv_file.write_text(csv_content)
        
        result = read_csv(str(csv_file))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_read_csv_nonexistent_file(self):
        """Test CSV reading with nonexistent file."""
        with pytest.raises(DataAccessError, match="Failed to read CSV file"):
            read_csv("/nonexistent/file.csv")

    def test_read_csv_invalid_content(self, tmp_path):
        """Test CSV reading with invalid content."""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text("invalid csv content\nwith\nmismatched\ncolumns,too,many")
        
        # Should still work as pandas is flexible, but test the error path
        with patch('pandas.read_csv', side_effect=Exception("Parse error")):
            with pytest.raises(DataAccessError, match="Failed to read CSV file"):
                read_csv(csv_file)


class TestWriteCsv:
    """Test write_csv function."""

    def test_write_csv_success(self, tmp_path):
        """Test successful CSV writing."""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'age': [25, 30],
            'city': ['NYC', 'LA']
        })
        csv_file = tmp_path / "output.csv"
        
        write_csv(df, csv_file)
        
        assert csv_file.exists()
        
        # Read back and verify
        result = pd.read_csv(csv_file, index_col=0)
        assert len(result) == 2
        assert list(result.columns) == ['name', 'age', 'city']

    def test_write_csv_create_directory(self, tmp_path):
        """Test CSV writing with directory creation."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        csv_file = tmp_path / "new_dir" / "output.csv"
        
        write_csv(df, csv_file)
        
        assert csv_file.exists()
        assert csv_file.parent.exists()

    def test_write_csv_with_kwargs(self, tmp_path):
        """Test CSV writing with additional arguments."""
        df = pd.DataFrame({'name': ['Alice'], 'age': [25]})
        csv_file = tmp_path / "output.csv"
        
        write_csv(df, csv_file, sep=';', index=False)
        
        assert csv_file.exists()
        content = csv_file.read_text()
        assert ';' in content
        assert '0' not in content  # No index

    def test_write_csv_string_path(self, tmp_path):
        """Test CSV writing with string path."""
        df = pd.DataFrame({'col': [1, 2]})
        csv_file = str(tmp_path / "output.csv")
        
        write_csv(df, csv_file)
        
        assert Path(csv_file).exists()

    def test_write_csv_permission_error(self, tmp_path):
        """Test CSV writing with permission error."""
        df = pd.DataFrame({'col': [1]})
        csv_file = tmp_path / "output.csv"
        
        with patch('pandas.DataFrame.to_csv', side_effect=OSError("Permission denied")):
            with pytest.raises(DataAccessError, match="Failed to write CSV file"):
                write_csv(df, csv_file)


class TestReadJson:
    """Test read_json function."""

    def test_read_json_success(self, tmp_path):
        """Test successful JSON reading."""
        json_file = tmp_path / "test.json"
        data = {"name": "Alice", "age": 25, "hobbies": ["reading", "coding"]}
        json_file.write_text(json.dumps(data))
        
        result = read_json(json_file)
        
        assert result == data

    def test_read_json_string_path(self, tmp_path):
        """Test JSON reading with string path."""
        json_file = tmp_path / "test.json"
        data = {"key": "value"}
        json_file.write_text(json.dumps(data))
        
        result = read_json(str(json_file))
        
        assert result == data

    def test_read_json_nonexistent_file(self):
        """Test JSON reading with nonexistent file."""
        with pytest.raises(DataAccessError, match="Failed to read JSON file"):
            read_json("/nonexistent/file.json")

    def test_read_json_invalid_content(self, tmp_path):
        """Test JSON reading with invalid JSON content."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("invalid json content")
        
        with pytest.raises(DataAccessError, match="Failed to read JSON file"):
            read_json(json_file)

    def test_read_json_empty_file(self, tmp_path):
        """Test JSON reading with empty file."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("")
        
        with pytest.raises(DataAccessError, match="Failed to read JSON file"):
            read_json(json_file)


class TestWriteJson:
    """Test write_json function."""

    def test_write_json_success(self, tmp_path):
        """Test successful JSON writing."""
        data = {"name": "Alice", "age": 25, "active": True}
        json_file = tmp_path / "output.json"
        
        write_json(data, json_file)
        
        assert json_file.exists()
        
        # Read back and verify
        with open(json_file, 'r') as f:
            result = json.load(f)
        assert result == data

    def test_write_json_create_directory(self, tmp_path):
        """Test JSON writing with directory creation."""
        data = {"key": "value"}
        json_file = tmp_path / "new_dir" / "output.json"
        
        write_json(data, json_file)
        
        assert json_file.exists()
        assert json_file.parent.exists()

    def test_write_json_custom_indent(self, tmp_path):
        """Test JSON writing with custom indentation."""
        data = {"nested": {"key": "value"}}
        json_file = tmp_path / "output.json"
        
        write_json(data, json_file, indent=4)
        
        assert json_file.exists()
        content = json_file.read_text()
        assert '    ' in content  # 4-space indentation

    def test_write_json_string_path(self, tmp_path):
        """Test JSON writing with string path."""
        data = {"test": True}
        json_file = str(tmp_path / "output.json")
        
        write_json(data, json_file)
        
        assert Path(json_file).exists()

    def test_write_json_permission_error(self, tmp_path):
        """Test JSON writing with permission error."""
        data = {"key": "value"}
        json_file = tmp_path / "output.json"
        
        with patch('builtins.open', side_effect=OSError("Permission denied")):
            with pytest.raises(DataAccessError, match="Failed to write JSON file"):
                write_json(data, json_file)


class TestProcessFilesInDirectory:
    """Test process_files_in_directory function."""

    def test_process_files_success(self, tmp_path):
        """Test successful file processing."""
        # Create test CSV files
        (tmp_path / "file1.csv").write_text("col1,col2\n1,2\n3,4")
        (tmp_path / "file2.csv").write_text("col1,col2\n5,6")
        (tmp_path / "file3.txt").write_text("not csv")
        
        def count_rows(file_path):
            df = pd.read_csv(file_path)
            return len(df)
        
        result = process_files_in_directory(tmp_path, count_rows, "*.csv")
        
        assert len(result) == 2
        assert result["file1.csv"] == 2
        assert result["file2.csv"] == 1
        assert "file3.txt" not in result

    def test_process_files_empty_directory(self, tmp_path):
        """Test processing empty directory."""
        def dummy_process(file_path):
            return "processed"
        
        result = process_files_in_directory(tmp_path, dummy_process)
        
        assert result == {}

    def test_process_files_no_matches(self, tmp_path):
        """Test with no matching files."""
        (tmp_path / "file.txt").touch()
        
        def dummy_process(file_path):
            return "processed"
        
        result = process_files_in_directory(tmp_path, dummy_process, "*.csv")
        
        assert result == {}

    def test_process_files_with_errors_logged(self, tmp_path):
        """Test file processing with errors (logged, not raised)."""
        (tmp_path / "good.csv").write_text("col1\n1\n2")
        (tmp_path / "bad.csv").write_text("invalid")
        
        def process_func(file_path):
            if "bad" in file_path.name:
                raise ValueError("Processing error")
            df = pd.read_csv(file_path)
            return len(df)
        
        with patch('metaquest.data.file_io.logger') as mock_logger:
            result = process_files_in_directory(tmp_path, process_func, "*.csv", raise_errors=False)
        
        assert len(result) == 1
        assert result["good.csv"] == 2
        assert "bad.csv" not in result
        mock_logger.error.assert_called()

    def test_process_files_with_errors_raised(self, tmp_path):
        """Test file processing with errors raised."""
        (tmp_path / "error.csv").write_text("invalid")
        
        def error_func(file_path):
            raise ValueError("Processing error")
        
        with pytest.raises(DataAccessError, match="Error processing"):
            process_files_in_directory(tmp_path, error_func, "*.csv", raise_errors=True)

    def test_process_files_string_path(self, tmp_path):
        """Test with string directory path."""
        (tmp_path / "test.csv").write_text("col\n1")
        
        def count_rows(file_path):
            df = pd.read_csv(file_path)
            return len(df)
        
        result = process_files_in_directory(str(tmp_path), count_rows, "*.csv")
        
        assert len(result) == 1
        assert result["test.csv"] == 1

    def test_process_files_directory_error(self):
        """Test with directory access error."""
        def dummy_process(file_path):
            return "processed"
        
        with patch('metaquest.data.file_io.list_files', side_effect=Exception("Access error")):
            result = process_files_in_directory("/invalid/path", dummy_process, raise_errors=False)
        
        assert result == {}

    def test_process_files_directory_error_raised(self):
        """Test with directory access error raised."""
        def dummy_process(file_path):
            return "processed"
        
        with patch('metaquest.data.file_io.list_files', side_effect=Exception("Access error")):
            with pytest.raises(DataAccessError, match="Error processing files"):
                process_files_in_directory("/invalid/path", dummy_process, raise_errors=True)


class TestFileIoIntegration:
    """Integration tests for file I/O functionality."""

    def test_csv_roundtrip(self, tmp_path):
        """Test complete CSV read/write roundtrip."""
        original_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'score': [95.5, 87.2, 92.1]
        })
        csv_file = tmp_path / "roundtrip.csv"
        
        # Write then read
        write_csv(original_df, csv_file, index=False)
        result_df = read_csv(csv_file)
        
        # Compare (pandas may change dtypes slightly)
        assert len(result_df) == len(original_df)
        assert list(result_df.columns) == list(original_df.columns)
        assert result_df['name'].tolist() == original_df['name'].tolist()

    def test_json_roundtrip(self, tmp_path):
        """Test complete JSON read/write roundtrip."""
        original_data = {
            "project": "MetaQuest",
            "version": "1.0.0",
            "features": ["testing", "analysis"],
            "config": {
                "threads": 4,
                "timeout": 30.5,
                "debug": True
            }
        }
        json_file = tmp_path / "roundtrip.json"
        
        # Write then read
        write_json(original_data, json_file)
        result_data = read_json(json_file)
        
        assert result_data == original_data

    def test_file_processing_workflow(self, tmp_path):
        """Test complete file processing workflow."""
        # Create directory structure
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"
        ensure_directory(data_dir)
        ensure_directory(output_dir)
        
        # Create sample CSV files
        for i in range(3):
            df = pd.DataFrame({
                'sample': [f'sample_{j}' for j in range(5)],
                'value': list(range(i*5, (i+1)*5))
            })
            write_csv(df, data_dir / f"dataset_{i}.csv", index=False)
        
        # Process files and collect statistics
        def get_stats(file_path):
            df = read_csv(file_path)
            return {
                'row_count': len(df),
                'mean_value': float(df['value'].mean()),
                'max_value': int(df['value'].max())
            }
        
        stats = process_files_in_directory(data_dir, get_stats, "*.csv")
        
        # Verify results
        assert len(stats) == 3
        for filename, file_stats in stats.items():
            assert file_stats['row_count'] == 5
            assert isinstance(file_stats['mean_value'], (int, float)) or hasattr(file_stats['mean_value'], 'item')
            assert isinstance(file_stats['max_value'], (int, float)) or hasattr(file_stats['max_value'], 'item')
        
        # Save summary
        summary_file = output_dir / "summary.json"
        write_json(stats, summary_file)
        
        # Verify summary can be read back
        loaded_stats = read_json(summary_file)
        assert loaded_stats == stats


if __name__ == "__main__":
    pytest.main([__file__])