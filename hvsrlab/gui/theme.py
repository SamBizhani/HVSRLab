"""One place that decides what the application looks like.

Qt 5.9 is the constraint here — no ``QPalette.PlaceholderText``, no
``:has()`` selectors, no stylesheet variables — so the palette is set through
``QPalette`` and the detail through one stylesheet built from the colours
below. Matplotlib gets the same colours, which is what stops the embedded
figures from looking pasted on.
"""

from __future__ import annotations

from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QApplication

#: The whole palette. Change it here and both Qt and matplotlib follow.
C = {
    "bg": "#11151b",           # window
    "surface": "#181d25",      # cards, panels
    "surface_alt": "#1f252f",  # inputs, hovered rows
    "border": "#2a323e",
    "border_soft": "#222933",

    "text": "#dfe6ee",
    "text_dim": "#8b98a8",
    "text_faint": "#5d6875",

    "accent": "#38bdf8",       # primary — selection, active nav, links
    "accent_dim": "#1d5c7a",
    "accent_text": "#04121b",

    "pick": "#fbbf24",         # the f0 pick, everywhere it appears
    "good": "#34d399",
    "warn": "#fbbf24",
    "bad": "#f87171",
    "muted": "#64748b",

    "series": [
        "#38bdf8", "#fbbf24", "#34d399", "#f472b6",
        "#a78bfa", "#fb923c", "#22d3ee", "#facc15",
    ],
}

#: Component colours, fixed so Z is always the same colour in every figure.
COMPONENT_COLORS = {"Z": "#38bdf8", "N": "#34d399", "E": "#fbbf24"}


def apply(app: QApplication, base_font_pt: int = 10) -> None:
    """Apply the dark palette, the stylesheet and the base font to *app*."""
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(C["bg"]))
    palette.setColor(QPalette.WindowText, QColor(C["text"]))
    palette.setColor(QPalette.Base, QColor(C["surface"]))
    palette.setColor(QPalette.AlternateBase, QColor(C["surface_alt"]))
    palette.setColor(QPalette.ToolTipBase, QColor(C["surface_alt"]))
    palette.setColor(QPalette.ToolTipText, QColor(C["text"]))
    palette.setColor(QPalette.Text, QColor(C["text"]))
    palette.setColor(QPalette.Button, QColor(C["surface_alt"]))
    palette.setColor(QPalette.ButtonText, QColor(C["text"]))
    palette.setColor(QPalette.BrightText, QColor(C["bad"]))
    palette.setColor(QPalette.Highlight, QColor(C["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor(C["accent_text"]))
    palette.setColor(QPalette.Link, QColor(C["accent"]))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(C["text_faint"]))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(C["text_faint"]))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(C["text_faint"]))
    app.setPalette(palette)

    font = QFont(_ui_font_family(), base_font_pt)
    app.setFont(font)
    app.setStyleSheet(stylesheet())


def _ui_font_family() -> str:
    from PyQt5.QtGui import QFontDatabase

    available = set(QFontDatabase().families())
    for name in ("Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans"):
        if name in available:
            return name
    return "Sans Serif"


def mono_font(pt: int = 9) -> QFont:
    from PyQt5.QtGui import QFontDatabase

    available = set(QFontDatabase().families())
    for name in ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Courier New"):
        if name in available:
            return QFont(name, pt)
    font = QFont()
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(pt)
    return font


def stylesheet() -> str:
    return f"""
QWidget {{
    color: {C['text']};
    background: transparent;
}}
QMainWindow, QDialog {{ background: {C['bg']}; }}

QToolTip {{
    background: {C['surface_alt']};
    color: {C['text']};
    border: 1px solid {C['border']};
    padding: 5px 7px;
}}

/* -- cards and panels ------------------------------------------------- */
QFrame#Card {{
    background: {C['surface']};
    border: 1px solid {C['border_soft']};
    border-radius: 8px;
}}
QLabel#CardTitle {{
    color: {C['text_dim']};
    font-size: 10pt;
    font-weight: 600;
    padding: 0 0 2px 0;
}}
QLabel#Hint {{ color: {C['text_faint']}; font-size: 9pt; }}
QLabel#PageTitle {{ font-size: 17pt; font-weight: 600; }}
QLabel#PageSubtitle {{ color: {C['text_dim']}; font-size: 10pt; }}
QLabel#StatValue {{ font-size: 19pt; font-weight: 600; }}
QLabel#StatLabel {{ color: {C['text_faint']}; font-size: 8pt;
                    text-transform: uppercase; }}

/* -- navigation -------------------------------------------------------- */
QListWidget#Nav {{
    background: {C['surface']};
    border: none;
    border-right: 1px solid {C['border_soft']};
    outline: none;
    padding: 6px 6px;
}}
QListWidget#Nav::item {{
    padding: 9px 12px;
    margin: 2px 0;
    border-radius: 6px;
    color: {C['text_dim']};
}}
QListWidget#Nav::item:hover {{ background: {C['surface_alt']}; color: {C['text']}; }}
QListWidget#Nav::item:selected {{
    background: {C['accent_dim']};
    color: {C['text']};
    font-weight: 600;
}}
QListWidget#Nav::item:disabled {{ color: {C['text_faint']}; }}

/* -- inputs ------------------------------------------------------------ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit,
QDateTimeEdit {{
    background: {C['surface_alt']};
    border: 1px solid {C['border']};
    border-radius: 5px;
    padding: 4px 7px;
    selection-background-color: {C['accent']};
    selection-color: {C['accent_text']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QDateTimeEdit:focus {{ border: 1px solid {C['accent']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{ color: {C['text_faint']}; background: {C['surface']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {C['surface_alt']};
    border: 1px solid {C['border']};
    selection-background-color: {C['accent_dim']};
    outline: none;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 14px; }}

/* -- buttons ----------------------------------------------------------- */
QPushButton {{
    background: {C['surface_alt']};
    border: 1px solid {C['border']};
    border-radius: 5px;
    padding: 6px 13px;
}}
QPushButton:hover {{ border-color: {C['accent']}; }}
QPushButton:pressed {{ background: {C['border']}; }}
QPushButton:disabled {{ color: {C['text_faint']}; border-color: {C['border_soft']}; }}
QPushButton#Primary {{
    background: {C['accent']};
    color: {C['accent_text']};
    border: none;
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background: #5bc9fa; }}
QPushButton#Primary:disabled {{ background: {C['accent_dim']}; color: {C['text_faint']}; }}
QPushButton#Danger {{ border-color: {C['bad']}; color: {C['bad']}; }}
QPushButton#Ghost {{ background: transparent; border: none; color: {C['text_dim']};
                     padding: 4px 8px; }}
QPushButton#Ghost:hover {{ color: {C['accent']}; }}
QPushButton:checked {{ background: {C['accent_dim']}; border-color: {C['accent']}; }}

/* -- tables ------------------------------------------------------------ */
QTableWidget, QTableView, QTreeWidget {{
    background: {C['surface']};
    alternate-background-color: {C['surface_alt']};
    border: 1px solid {C['border_soft']};
    border-radius: 6px;
    gridline-color: {C['border_soft']};
    selection-background-color: {C['accent_dim']};
    selection-color: {C['text']};
    outline: none;
}}
QHeaderView::section {{
    background: {C['surface_alt']};
    color: {C['text_dim']};
    border: none;
    border-right: 1px solid {C['border_soft']};
    border-bottom: 1px solid {C['border']};
    padding: 5px 7px;
    font-weight: 600;
}}
QTableWidget::item {{ padding: 3px 4px; }}

/* -- misc -------------------------------------------------------------- */
QProgressBar {{
    background: {C['surface_alt']};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 3px; }}

QSplitter::handle {{ background: {C['border_soft']}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 5px;
                               min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {C['muted']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {C['border']}; border-radius: 5px;
                                 min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QTabWidget::pane {{ border: 1px solid {C['border_soft']}; border-radius: 6px;
                    top: -1px; }}
QTabBar::tab {{
    background: transparent;
    color: {C['text_dim']};
    padding: 6px 14px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {C['text']}; border-bottom: 2px solid {C['accent']}; }}
QTabBar::tab:hover {{ color: {C['text']}; }}

QGroupBox {{
    border: 1px solid {C['border_soft']};
    border-radius: 6px;
    margin-top: 9px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 9px;
    color: {C['text_dim']};
    font-weight: 600;
}}

QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
QCheckBox::indicator:unchecked {{
    border: 1px solid {C['border']}; border-radius: 3px; background: {C['surface_alt']};
}}
QCheckBox::indicator:checked {{
    border: 1px solid {C['accent']}; border-radius: 3px; background: {C['accent']};
}}

QSlider::groove:horizontal {{ height: 4px; background: {C['border']};
                              border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {C['accent']}; width: 13px;
                              margin: -5px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {C['accent_dim']}; border-radius: 2px; }}

QStatusBar {{ background: {C['surface']}; border-top: 1px solid {C['border_soft']};
              color: {C['text_dim']}; }}
QStatusBar::item {{ border: none; }}

QMenuBar {{ background: {C['surface']}; border-bottom: 1px solid {C['border_soft']}; }}
QMenuBar::item {{ padding: 5px 11px; background: transparent; }}
QMenuBar::item:selected {{ background: {C['surface_alt']}; }}
QMenu {{ background: {C['surface_alt']}; border: 1px solid {C['border']};
         padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {C['accent_dim']}; }}
QMenu::separator {{ height: 1px; background: {C['border']}; margin: 4px 8px; }}

QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{ background: {C['surface_alt']}; padding: 5px 9px;
                      border-bottom: 1px solid {C['border']}; }}
"""


def mpl_rc() -> dict:
    """Matplotlib rcParams that match the Qt theme."""
    return {
        "figure.facecolor": C["surface"],
        "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"],
        "axes.edgecolor": C["border"],
        "axes.labelcolor": C["text_dim"],
        "axes.titlecolor": C["text"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C["border_soft"],
        "grid.linewidth": 0.7,
        "text.color": C["text"],
        "xtick.color": C["text_dim"],
        "ytick.color": C["text_dim"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
        "figure.autolayout": False,
    }


def install_mpl() -> None:
    import matplotlib

    matplotlib.rcParams.update(mpl_rc())
