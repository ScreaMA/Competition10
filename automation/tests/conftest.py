"""自动化系统测试的公共配置：把 automation/ 加入 sys.path。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTOMATION_DIR = ROOT / "automation"
if str(AUTOMATION_DIR) not in sys.path:
    sys.path.insert(0, str(AUTOMATION_DIR))
