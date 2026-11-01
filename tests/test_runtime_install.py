"""Exercise the dependency filter without installing a runtime or importing CUDA."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_only_unused_fastapi_cli_extra_is_removed():
    path=ROOT/'scripts/install_runtime.py'
    spec=importlib.util.spec_from_file_location('runtime_install',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows=['fastapi[standard]>=0.115.0','fastapi-cli>=0.0.5','fastapi-cloud-cli>=0.1',
          'torch==2.8.0','uvicorn[standard]>=0.30','python-multipart>=0.0.18',
          'email-validator>=2.0','transformers>=4.55.2']
    result=module.requirements_without_cli(rows)
    assert 'fastapi>=0.115.0' in result
    assert not any('fastapi-cli' in row or 'fastapi-cloud-cli' in row for row in result)
    for row in rows[3:]:
        assert row in result
    assert 'configs/requirements.txt' in path.read_text(encoding='utf-8')
    assert 'configs/requirements-opsd.txt' not in path.read_text(encoding='utf-8')
