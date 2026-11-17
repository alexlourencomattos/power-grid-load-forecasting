import os
import tempfile
import unittest
from pathlib import Path

from ons_load_pipeline.pipeline import _acquire_lock


class PipelineLockTests(unittest.TestCase):
    def test_rejects_concurrent_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            descriptor, lock_path = _acquire_lock(Path(directory))
            try:
                with self.assertRaisesRegex(RuntimeError, "Outra execução"):
                    _acquire_lock(Path(directory))
            finally:
                os.close(descriptor)
                lock_path.unlink()


if __name__ == "__main__":
    unittest.main()
