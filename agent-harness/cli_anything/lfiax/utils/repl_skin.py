"""cli-anything REPL Skin — Unified terminal interface for all CLI harnesses.

Copy this file into your CLI package at:
    cli_anything/<software>/utils/repl_skin.py

Usage:
    from cli_anything.<software>.utils.repl_skin import ReplSkin

    skin = ReplSkin("lfiax", version="1.0.0")
    skin.print_banner()
    prompt_text = skin.prompt(project_name="my_experiment", modified=True)
    skin.success("Experiment completed")
    skin.error("Config not found")
"""

import os
import sys

# -- ANSI color codes --

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_UNDERLINE = "\033[4m"

_CYAN = "\033[38;5;80m"
_CYAN_BG = "\033[48;5;80m"
_WHITE = "\033[97m"
_GRAY = "\033[38;5;245m"
_DARK_GRAY = "\033[38;5;240m"
_LIGHT_GRAY = "\033[38;5;250m"

_ACCENT_COLORS = {
    "gimp":        "\033[38;5;214m",
    "blender":     "\033[38;5;208m",
    "inkscape":    "\033[38;5;39m",
    "audacity":    "\033[38;5;33m",
    "libreoffice": "\033[38;5;40m",
    "obs_studio":  "\033[38;5;55m",
    "kdenlive":    "\033[38;5;69m",
    "shotcut":     "\033[38;5;35m",
    "lfiax":       "\033[38;5;117m",
}
_DEFAULT_ACCENT = "\033[38;5;75m"

_GREEN = "\033[38;5;78m"
_YELLOW = "\033[38;5;220m"
_RED = "\033[38;5;196m"
_BLUE = "\033[38;5;75m"
_MAGENTA = "\033[38;5;176m"

_ICON = f"{_CYAN}{_BOLD}\u25c6{_RESET}"
_ICON_SMALL = f"{_CYAN}\u25b8{_RESET}"

_H_LINE = "\u2500"
_V_LINE = "\u2502"
_TL = "\u256d"
_TR = "\u256e"
_BL = "\u2570"
_BR = "\u256f"
_T_DOWN = "\u252c"
_T_UP = "\u2534"
_T_RIGHT = "\u251c"
_T_LEFT = "\u2524"
_CROSS = "\u253c"


def _strip_ansi(text: str) -> str:
    import re
    return re.sub(r"\033\[[^m]*m", "", text)


def _visible_len(text: str) -> int:
    return len(_strip_ansi(text))


class ReplSkin:
    """Unified REPL skin for cli-anything CLIs."""

    def __init__(self, software: str, version: str = "1.0.0",
                 history_file: str | None = None, skill_path: str | None = None):
        self.software = software.lower().replace("-", "_")
        self.display_name = software.replace("_", " ").title()
        self.version = version

        if skill_path is None:
            from pathlib import Path
            _auto = Path(__file__).resolve().parent.parent / "skills" / "SKILL.md"
            if _auto.is_file():
                skill_path = str(_auto)
        self.skill_path = skill_path
        self.accent = _ACCENT_COLORS.get(self.software, _DEFAULT_ACCENT)

        if history_file is None:
            from pathlib import Path
            hist_dir = Path.home() / f".cli-anything-{self.software}"
            hist_dir.mkdir(parents=True, exist_ok=True)
            self.history_file = str(hist_dir / "history")
        else:
            self.history_file = history_file

        self._color = self._detect_color_support()

    def _detect_color_support(self) -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("CLI_ANYTHING_NO_COLOR"):
            return False
        if not hasattr(sys.stdout, "isatty"):
            return False
        return sys.stdout.isatty()

    def _c(self, code: str, text: str) -> str:
        if not self._color:
            return text
        return f"{code}{text}{_RESET}"

    # -- Banner --

    def print_banner(self):
        inner = 54

        def _box_line(content: str) -> str:
            pad = inner - _visible_len(content)
            vl = self._c(_DARK_GRAY, _V_LINE)
            return f"{vl}{content}{' ' * max(0, pad)}{vl}"

        top = self._c(_DARK_GRAY, f"{_TL}{_H_LINE * inner}{_TR}")
        bot = self._c(_DARK_GRAY, f"{_BL}{_H_LINE * inner}{_BR}")

        icon = self._c(_CYAN + _BOLD, "\u25c6")
        brand = self._c(_CYAN + _BOLD, "cli-anything")
        dot = self._c(_DARK_GRAY, "\u00b7")
        name = self._c(self.accent + _BOLD, self.display_name)
        title = f" {icon}  {brand} {dot} {name}"

        ver = f" {self._c(_DARK_GRAY, f'   v{self.version}')}"
        tip = f" {self._c(_DARK_GRAY, '   Type help for commands, quit to exit')}"
        empty = ""

        skill_line = None
        if self.skill_path:
            skill_icon = self._c(_MAGENTA, "\u25c7")
            skill_label = self._c(_DARK_GRAY, "   Skill:")
            skill_path_display = self._c(_LIGHT_GRAY, self.skill_path)
            skill_line = f" {skill_icon} {skill_label} {skill_path_display}"

        print(top)
        print(_box_line(title))
        print(_box_line(ver))
        if skill_line:
            print(_box_line(skill_line))
        print(_box_line(empty))
        print(_box_line(tip))
        print(bot)
        print()

    # -- Prompt --

    def prompt(self, project_name: str = "", modified: bool = False,
               context: str = "") -> str:
        parts = []
        if self._color:
            parts.append(f"{_CYAN}\u25c6{_RESET} ")
        else:
            parts.append("> ")
        parts.append(self._c(self.accent + _BOLD, self.software))
        if project_name or context:
            ctx = context or project_name
            mod = "*" if modified else ""
            parts.append(f" {self._c(_DARK_GRAY, '[')}")
            parts.append(self._c(_LIGHT_GRAY, f"{ctx}{mod}"))
            parts.append(self._c(_DARK_GRAY, ']'))
        parts.append(self._c(_GRAY, " \u276f "))
        return "".join(parts)

    def prompt_tokens(self, project_name: str = "", modified: bool = False,
                      context: str = ""):
        tokens = []
        tokens.append(("class:icon", "\u25c6 "))
        tokens.append(("class:software", self.software))
        if project_name or context:
            ctx = context or project_name
            mod = "*" if modified else ""
            tokens.append(("class:bracket", " ["))
            tokens.append(("class:context", f"{ctx}{mod}"))
            tokens.append(("class:bracket", "]"))
        tokens.append(("class:arrow", " \u276f "))
        return tokens

    def get_prompt_style(self):
        try:
            from prompt_toolkit.styles import Style
        except ImportError:
            return None
        accent_hex = _ANSI_256_TO_HEX.get(self.accent, "#5fafff")
        return Style.from_dict({
            "icon": "#5fdfdf bold",
            "software": f"{accent_hex} bold",
            "bracket": "#585858",
            "context": "#bcbcbc",
            "arrow": "#808080",
            "completion-menu.completion": "bg:#303030 #bcbcbc",
            "completion-menu.completion.current": f"bg:{accent_hex} #000000",
            "completion-menu.meta.completion": "bg:#303030 #808080",
            "completion-menu.meta.completion.current": f"bg:{accent_hex} #000000",
            "auto-suggest": "#585858",
            "bottom-toolbar": "bg:#1c1c1c #808080",
            "bottom-toolbar.text": "#808080",
        })

    # -- Messages --

    def success(self, message: str):
        icon = self._c(_GREEN + _BOLD, "\u2713")
        print(f"  {icon} {self._c(_GREEN, message)}")

    def error(self, message: str):
        icon = self._c(_RED + _BOLD, "\u2717")
        print(f"  {icon} {self._c(_RED, message)}", file=sys.stderr)

    def warning(self, message: str):
        icon = self._c(_YELLOW + _BOLD, "\u26a0")
        print(f"  {icon} {self._c(_YELLOW, message)}")

    def info(self, message: str):
        icon = self._c(_BLUE, "\u25cf")
        print(f"  {icon} {self._c(_LIGHT_GRAY, message)}")

    def hint(self, message: str):
        print(f"  {self._c(_DARK_GRAY, message)}")

    def section(self, title: str):
        print()
        print(f"  {self._c(self.accent + _BOLD, title)}")
        print(f"  {self._c(_DARK_GRAY, _H_LINE * len(title))}")

    # -- Status --

    def status(self, label: str, value: str):
        lbl = self._c(_GRAY, f"  {label}:")
        val = self._c(_WHITE, f" {value}")
        print(f"{lbl}{val}")

    def status_block(self, items: dict, title: str = ""):
        if title:
            self.section(title)
        if not isinstance(items, dict):
            return
        max_key = max(len(str(k)) for k in items) if items else 0
        for label, value in items.items():
            lbl = self._c(_GRAY, f"  {str(label):<{max_key}}")
            val = self._c(_WHITE, f"  {value}")
            print(f"{lbl}{val}")

    def progress(self, current: int, total: int, label: str = ""):
        pct = int(current / total * 100) if total > 0 else 0
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        text = f"  {self._c(_CYAN, bar)} {self._c(_GRAY, f'{pct:3d}%')}"
        if label:
            text += f" {self._c(_LIGHT_GRAY, label)}"
        print(text)

    # -- Table --

    def table(self, headers: list, rows: list, max_col_width: int = 40):
        if not headers:
            return
        col_widths = [min(len(h), max_col_width) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = min(
                        max(col_widths[i], len(str(cell))), max_col_width
                    )

        def pad(text, width):
            t = str(text)[:width]
            return t + " " * (width - len(t))

        header_cells = [
            self._c(_CYAN + _BOLD, pad(h, col_widths[i]))
            for i, h in enumerate(headers)
        ]
        sep = self._c(_DARK_GRAY, f" {_V_LINE} ")
        print(f"  {sep.join(header_cells)}")

        sep_line = self._c(
            _DARK_GRAY,
            f"  {'\u2500\u2500\u2500'.join([_H_LINE * w for w in col_widths])}"
        )
        print(sep_line)

        for row in rows:
            cells = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cells.append(self._c(_LIGHT_GRAY, pad(str(cell), col_widths[i])))
            row_sep = self._c(_DARK_GRAY, f" {_V_LINE} ")
            print(f"  {row_sep.join(cells)}")

    # -- Help --

    def help(self, commands: dict):
        self.section("Commands")
        max_cmd = max(len(c) for c in commands) if commands else 0
        for cmd, desc in commands.items():
            cmd_styled = self._c(self.accent, f"  {cmd:<{max_cmd}}")
            desc_styled = self._c(_GRAY, f"  {desc}")
            print(f"{cmd_styled}{desc_styled}")
        print()

    # -- Goodbye --

    def print_goodbye(self):
        print(f"\n  {_ICON_SMALL} {self._c(_GRAY, 'Goodbye!')}\n")

    # -- Prompt toolkit session --

    def create_prompt_session(self):
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import FileHistory
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

            style = self.get_prompt_style()
            session = PromptSession(
                history=FileHistory(self.history_file),
                auto_suggest=AutoSuggestFromHistory(),
                style=style,
                enable_history_search=True,
            )
            return session
        except ImportError:
            return None

    def get_input(self, pt_session, project_name: str = "",
                  modified: bool = False, context: str = "") -> str:
        if pt_session is not None:
            from prompt_toolkit.formatted_text import FormattedText
            tokens = self.prompt_tokens(project_name, modified, context)
            return pt_session.prompt(FormattedText(tokens)).strip()
        else:
            raw_prompt = self.prompt(project_name, modified, context)
            return input(raw_prompt).strip()

    def bottom_toolbar(self, items: dict):
        def toolbar():
            from prompt_toolkit.formatted_text import FormattedText
            parts = []
            for i, (k, v) in enumerate(items.items()):
                if i > 0:
                    parts.append(("class:bottom-toolbar.text", "  \u2502  "))
                parts.append(("class:bottom-toolbar.text", f" {k}: "))
                parts.append(("class:bottom-toolbar", v))
            return FormattedText(parts)
        return toolbar


_ANSI_256_TO_HEX = {
    "\033[38;5;33m":  "#0087ff",
    "\033[38;5;35m":  "#00af5f",
    "\033[38;5;39m":  "#00afff",
    "\033[38;5;40m":  "#00d700",
    "\033[38;5;55m":  "#5f00af",
    "\033[38;5;69m":  "#5f87ff",
    "\033[38;5;75m":  "#5fafff",
    "\033[38;5;80m":  "#5fd7d7",
    "\033[38;5;117m": "#87d7ff",
    "\033[38;5;208m": "#ff8700",
    "\033[38;5;214m": "#ffaf00",
}
