import importlib
import sys


def test_support_import_does_not_load_solver_modules():
    sys.modules.pop("anchorplate", None)
    sys.modules.pop("anchorplate.solver", None)
    sys.modules.pop("anchorplate.plotting", None)

    importlib.import_module("anchorplate.support")

    assert "anchorplate.solver" not in sys.modules
    assert "anchorplate.plotting" not in sys.modules


def test_model_import_does_not_load_solver_modules():
    sys.modules.pop("anchorplate", None)
    sys.modules.pop("anchorplate.solver", None)
    sys.modules.pop("anchorplate.plotting", None)

    importlib.import_module("anchorplate.model")

    assert "anchorplate.solver" not in sys.modules
    assert "anchorplate.plotting" not in sys.modules
