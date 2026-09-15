"""把任务环境落成目录：复刻 `/tmp/selfEvolutionTask/…`。

目录结构（设计文档V2 §6.2 实测）：

    <root>/1-fixed-step/1-unknown-api/
        task_1_beijing.md      ← 题面（27 字节的 phaseTask 指向它）
        task_N_<城市>.md       ← 同族、不同参数 ⇒ 技能复用的主战场
        API_DOCS.md            ← **故意过时**的文档（题眼）
    <root>/1-fixed-step/2-engineering-fix/
        task_1_<代号>.md
        ws_1/
            spec.md            ← 修复清单
            check              ← CRLF + 无执行位（跑起来就是 bad interpreter）
            <变体声明的工作区>  ← 缺失目录 / 错配置 / 缺脚本 / 无权限

**一次只建一个工程修复变体**（`ws_1` 是 `locate()` 认的那个名字）。要换题就
重建一份沙盒——构建是几十毫秒的事，比想办法让两套工作区共存省心。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import fix_tasks
from .fix_tasks import FixVariant, KNOWN_VARIANT
from .apiserver import CITY_SLUG


class Sandbox:
    """一份落盘的任务环境"""

    def __init__(self, root: Path, base_url: str):
        self.root = Path(root)
        self.base_url = base_url.rstrip("/")
        #: 城市 -> 题面编号（`_build_api_tasks` 填）
        self.api_index: dict[str, int] = {}

    # --- 路径 ---

    @property
    def api_dir(self) -> Path:
        return self.root / "1-fixed-step" / "1-unknown-api"

    @property
    def ws_root(self) -> Path:
        return self.root / "1-fixed-step" / "2-engineering-fix"

    @property
    def ws(self) -> Path:
        return self.ws_root / "ws_1"

    @property
    def spec(self) -> Path:
        return self.ws / "spec.md"

    @property
    def check(self) -> Path:
        return self.ws / "check"

    def task_file(self, city: str, index: int) -> Path:
        return self.api_dir / ("task_%d_%s.md" % (index, CITY_SLUG[city]))

    def api_phase(self, city: str) -> str:
        """这一族任务的 `phaseTask` 原文（27 字节、零关键词）

        必须和**实际建出来的文件名**对得上：`task_hint` 是从这句话里抠出来的，
        对不上就退化成"扫目录找 task_*.md"，测的就不是生产路径了。
        """
        return "请阅读task_%d_%s.md，获取任务信息" % (
            self.api_index[city], CITY_SLUG[city]
        )

    # --- 构建 ---

    def build(
        self,
        *,
        cities: tuple[str, ...] = ("北京",),
        variant: FixVariant = KNOWN_VARIANT,
    ) -> "Sandbox":
        self.api_dir.mkdir(parents=True, exist_ok=True)
        (self.ws / "bin").mkdir(parents=True, exist_ok=True)

        self._build_api_tasks(cities)
        self._build_fix_task(variant)
        return self

    def _build_api_tasks(self, cities: tuple[str, ...]) -> None:
        from . import fixtures

        for index, city in enumerate(cities, start=1):
            self.api_index[city] = index
            (self.api_dir / ("task_%d_%s.md" % (index, CITY_SLUG[city]))).write_text(
                fixtures.heritage_task(city, index).replace(
                    "http://localhost:8899", self.base_url
                ),
                encoding="utf-8",
            )
        # `API_DOCS.md` 的 `基础URL` 指向真实端口；**鉴权头与参数名保持文档原样**
        # ——那份"过时"正是要考的东西，不能替它改对。
        (self.api_dir / "API_DOCS.md").write_text(
            fixtures.API_DOCS.replace("http://localhost:8899", self.base_url),
            encoding="utf-8",
        )

    def _build_fix_task(self, variant: FixVariant) -> None:
        ws_posix = "/tmp/selfEvolutionTask/1-fixed-step/2-engineering-fix/ws_1"
        (self.ws_root / ("task_1_%s.md" % variant.code)).write_text(
            fix_tasks.task_text(variant, ws_posix), encoding="utf-8"
        )
        self.spec.write_text(fix_tasks.spec_text(variant), encoding="utf-8")

        # 配置：先建好目录（含 `etc/` 这种不在 spec 里的层级）
        conf_path = self.ws / variant.conf
        conf_path.parent.mkdir(parents=True, exist_ok=True)
        conf_path.write_text(fix_tasks.conf_text(variant, correct=False), encoding="utf-8")

        # 目录：该缺的缺、该在的在（"在的那个"权限也先弄坏，才需要修）
        for folder in variant.dirs:
            path = self.ws / folder
            if folder in variant.missing_dirs:
                continue
            path.mkdir(parents=True, exist_ok=True)
            _degrade_perms(path)

        # 脚本：按变体坏成 crlf / missing / perm
        script_path = self.ws / variant.script
        if variant.script_broken != "missing":
            script_path.parent.mkdir(parents=True, exist_ok=True)
            _write_crlf(script_path, fix_tasks.script_text(variant))
            if variant.script_broken == "perm":
                pass          # 只坏权限，`_write_crlf` 已经不带执行位
            else:
                script_path.chmod(0o644)

        # `check` 自己：CRLF + 无执行位（实测形态）
        _write_crlf(self.check, fix_tasks.check_text(variant))

    # --- 判据 ---

    def run_check(self) -> tuple[int, str]:
        """直接用 POSIX shell 跑 `./check`（绕过客户端的调用方式）

        客户端是通过 `sh -c` 跑它的，Windows 上那条路走不通（见 `runner.py`），
        所以判"环境修好了没有"要用这条独立通道。
        """
        shell = shutil.which("sh") or shutil.which("bash")
        if shell is None:
            return -1, "本机没有 POSIX shell，无法执行 ./check"
        proc = subprocess.run(
            [shell, "./check"], cwd=str(self.ws), timeout=15,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return proc.returncode, proc.stdout.decode("utf-8", "replace")

    def check_passes(self) -> bool:
        code, out = self.run_check()
        return code == 0 and "TOKEN" in out

    def plant_fix(self, variant: FixVariant = KNOWN_VARIANT) -> None:
        """把工作区修成"正确状态"——用来验证判据本身是可满足的

        连手工修好都过不了 `check`，那 `check` 就是坏的，后面所有"任务完不成"
        的结论都不成立。
        """
        for name in (variant.script, "check"):
            path = self.ws / name
            if not path.exists():
                # `script_broken="missing"` 的变体：要**建出来**
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    fix_tasks.script_text(variant) if name == variant.script
                    else fix_tasks.check_text(variant),
                    encoding="utf-8",
                )
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
            path.chmod(path.stat().st_mode | 0o111)
        conf = self.ws / variant.conf
        conf.write_text(fix_tasks.conf_text(variant, correct=True), encoding="utf-8")
        for folder in variant.dirs:
            path = self.ws / folder
            path.mkdir(parents=True, exist_ok=True)
            try:
                path.chmod(0o755)
            except OSError:
                pass


def _degrade_perms(path: Path) -> None:
    """把目录权限改坏（修好之前 `check` 的权限项应该报错）"""
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _write_crlf(path: Path, text: str) -> None:
    """按 CRLF 落盘，并去掉可执行位（`bad interpreter: /bin/sh^M` 的两个条件）"""
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    try:
        path.chmod(0o644)
    except OSError:
        pass


def build(root: Path, base_url: str, **kwargs) -> Sandbox:
    return Sandbox(root, base_url).build(**kwargs)
