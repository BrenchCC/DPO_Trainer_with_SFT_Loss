"""Tests for complete dataset validation."""

import os
import sys
import json
import tempfile
import unittest

sys.path.append(os.getcwd())

from utils.tools import validate_data_format


class ValidationTestCase(unittest.TestCase):
    """Verify validation checks every record and supported history forms."""

    def _validate(self, data, mode: str) -> bool:
        """Validate temporary JSON data.

        Args:
            data: JSON-serializable dataset records.
            mode: Validation mode.

        Returns:
            Validation result.
        """
        with tempfile.NamedTemporaryFile(
            mode = "w",
            suffix = ".json",
            encoding = "utf-8",
            delete = False
        ) as file:
            json.dump(data, file)
            temp_path = file.name

        try:
            return validate_data_format(temp_path, mode)
        finally:
            os.remove(temp_path)

    def test_validation_rejects_invalid_later_record(self) -> None:
        """Verify validation does not stop after checking the first record."""
        data = [
            {
                "instruction": "question",
                "chosen": "good",
                "rejected": "bad"
            },
            {
                "instruction": "question",
                "chosen": "good"
            }
        ]

        self.assertFalse(self._validate(data, "dpo"))

    def test_validation_accepts_system_only_history(self) -> None:
        """Verify role-message DPO history can contain a system prompt."""
        data = [
            {
                "instruction": "question",
                "chosen": "good",
                "rejected": "bad",
                "history": [{"role": "system", "content": "system"}]
            }
        ]

        self.assertTrue(self._validate(data, "dpo"))


if __name__ == "__main__":
    unittest.main()
