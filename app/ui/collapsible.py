from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A section widget with a clickable header that toggles its content's visibility.

    Usage:
        section = CollapsibleSection("Particle Removal")
        section.addWidget(my_checkbox)
        section.addLayout(my_hbox)
        parent_layout.addWidget(section)
    """

    HEADER_STYLE = (
        "QPushButton { text-align: left; padding: 6px 8px; font-weight: bold; "
        "color: #1a3a5c; background: #d8e4f0; border: none; border-radius: 3px; }"
        "QPushButton:hover { background: #c4d4ec; }"
    )

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self.header_btn = QPushButton()
        self.header_btn.setCheckable(True)
        self.header_btn.setChecked(expanded)
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_btn.setStyleSheet(self.HEADER_STYLE)
        self.header_btn.toggled.connect(self._on_toggled)
        outer.addWidget(self.header_btn)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 4, 4, 6)
        self.content_layout.setSpacing(4)
        outer.addWidget(self.content_widget)

        self.content_widget.setVisible(expanded)
        self._update_header()

    def _on_toggled(self, checked: bool):
        self.content_widget.setVisible(checked)
        self._update_header()

    def _update_header(self):
        arrow = "▼" if self.header_btn.isChecked() else "▶"
        self.header_btn.setText(f"{arrow}  {self._title}")

    # Forwarding helpers so the section looks like a layout to callers
    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

    def addSpacing(self, n: int):
        self.content_layout.addSpacing(n)
