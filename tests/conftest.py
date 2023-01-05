import os
import pytest

depends_on_maestro_dataset = pytest.mark.skipif(
    "MAESTRO_DATASET_DIR" not in os.environ.keys(),
    reason="Environment variable 'MAESTRO_DATASET_DIR' missing."
)
