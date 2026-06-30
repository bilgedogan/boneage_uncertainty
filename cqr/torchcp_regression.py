import sys
import types


for module_name in ["torchcp.classification", "torchcp.graph", "torchcp.llm"]:
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

from torchcp.regression.loss import QuantileLoss
from torchcp.regression.score import CQR


__all__ = ["QuantileLoss", "CQR"]
