# 调度脚本配置：数据文件路径与解析标记

from pathlib import Path

# ===============================
# 读取目标文件配置（保持在文件顶部，方便快速调整）
# ===============================

DEFAULT_DATA_DIR = Path(r"./")
DEFAULT_TARGET_FILE = DEFAULT_DATA_DIR / "test.mm"

# 文件分段标记
MARKERS = {"*", "**", "***", "****", "*****", "******"}
