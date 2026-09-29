import tempfile
import unittest
from pathlib import Path

from decision_use_case.config import load_experiment_config


class ConfigTest(unittest.TestCase):
    def test_load_config_resolves_paths_relative_to_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            config_path = tmp_path / "experiment.yaml"
            config_path.write_text(
                """
database_path: data/financial.sqlite
output_dir: runs
loan_ids: [4959]
model:
  name: qwen3:4b
  base_url: http://localhost:11434/v1
""",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

            self.assertEqual(config.database_path, (tmp_path / "data" / "financial.sqlite").resolve())
            self.assertEqual(config.output_dir, (tmp_path / "runs").resolve())
            self.assertEqual(config.model.name, "qwen3:4b")


if __name__ == "__main__":
    unittest.main()

