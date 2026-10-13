"""Room layout tab: draw the room, place the boards, line up a floor plan.

Edits host/links.json through the Qt-free model in layout.py. The canvas
works in meters with y pointing up, as the heatmap draws the room, so what
you place here is what the heatmap shows. Boards and corners are dragged
on the canvas or typed into the tables beside it; a floor-plan image can
be put underneath and scaled by clicking two points a known distance
apart.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF, QTransform,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from csidetector.provision import devices as dv
from csidetector.provision import layout as lo

PX_PER_M = 100.0          # view pixels per meter at zoom 1
SNAP_M = 0.1
TX_COLOR = QColor("#1c7eb6")
RX_COLOR = QColor("#e08a1e")


def _snap(v: float) -> float:
    return round(v / SNAP_M) * SNAP_M


# --------------------------------------------------------------------------
# Canvas items
# --------------------------------------------------------------------------

class _Handle(QGraphicsItem):
    """A marker of fixed pixel size at a point in meters. Ignores the view's
    zoom and y-flip, so it stays the same size and its label stays upright."""

    def __init__(self, canvas: "RoomCanvas", radius: float):
        super().__init__()
        self.canvas = canvas
        self.r = radius
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
                      | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def boundingRect(self) -> QRectF:  # noqa: N802
        r = self.r + 3
        return QRectF(-r, -r, 2 * r, 2 * r)

    def itemChange(self, change, value):  # noqa: N802
        if (change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
                and self.canvas.snap and not self.canvas.updating):
            return QPointF(_snap(value.x()), _snap(value.y()))
        if (change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
                and not self.canvas.updating):
            self.moved(value.x(), value.y())
        return super().itemChange(change, value)

    def moved(self, x: float, y: float) -> None:
        raise NotImplementedError


class _NodeItem(_Handle):
    def __init__(self, canvas: "RoomCanvas", node: lo.Node):
        super().__init__(canvas, 9)
        self.node = node
        self.setZValue(30)
        self.text = QGraphicsSimpleTextItem(self)
        self.text.setPos(self.r + 4, -self.r)
        self.setToolTip(f"{node.role.upper()} {node.label or ''} {node.mac}".strip())
        self.refresh()

    def refresh(self) -> None:
        n = self.node
        self.text.setText((n.label or n.mac) + ("" if n.placed else "  (not placed)"))
        self.update()

    def paint(self, p: QPainter, option, widget=None) -> None:
        color = TX_COLOR if self.node.role == dv.ROLE_TX else RX_COLOR
        pen = QPen(color.darker(160), 3 if self.isSelected() else 1.5)
        if not self.node.placed:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(QBrush(color if self.node.placed else color.lighter(170)))
        if self.node.role == dv.ROLE_TX:
            r = self.r
            p.drawPolygon(QPolygonF([QPointF(0, -r), QPointF(r, r * 0.8),
                                     QPointF(-r, r * 0.8)]))
        else:
            p.drawEllipse(QPointF(0, 0), self.r, self.r)

    def moved(self, x: float, y: float) -> None:
        self.canvas.model.move_node(self.node, x, y)
        self.refresh()
        self.canvas.edited.emit()


class _CornerItem(_Handle):
    def __init__(self, canvas: "RoomCanvas", index: int):
        super().__init__(canvas, 5)
        self.index = index
        self.setZValue(20)

    def paint(self, p: QPainter, option, widget=None) -> None:
        p.setPen(QPen(QColor("#333333"), 2 if self.isSelected() else 1))
        p.setBrush(QBrush(QColor("#ffffff")))
        p.drawRect(QRectF(-self.r, -self.r, 2 * self.r, 2 * self.r))

    def moved(self, x: float, y: float) -> None:
        self.canvas.model.move_corner(self.index, x, y)
        self.canvas.update_outline()
        self.canvas.edited.emit()


class _ImageItem(QGraphicsPixmapItem):
    """The floor plan; reports drags (Move image) so they count as edits."""

    def __init__(self, canvas: "RoomCanvas", pixmap: QPixmap):
        super().__init__(pixmap)
        self.canvas = canvas
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    def itemChange(self, change, value):  # noqa: N802
        if (change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
                and not self.canvas.updating):
            self.canvas.image_moved.emit()
        return super().itemChange(change, value)


# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------

class RoomCanvas(QGraphicsView):
    """The room in meters, y up, with a 0.5 m grid."""

    edited = pyqtSignal()                 # the layout changed on the canvas
    selection = pyqtSignal(object)        # the selected Node, or None
    hovered = pyqtSignal(float, float)    # mouse position in meters
    picked = pyqtSignal(float, float)     # a click while picking points
    image_moved = pyqtSignal()            # the floor plan was dragged

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAccessibleName("Room canvas")
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setMouseTracking(True)
        self.setMinimumSize(420, 320)
        self.setTransform(QTransform.fromScale(PX_PER_M, -PX_PER_M))
        self.model = lo.Layout()
        self.snap = True
        self.updating = False             # set while we move items ourselves
        self.picking = 0                  # clicks still wanted by pick_points()
        self._zoomed = False              # the user zoomed: stop auto-fitting
        self._image_item: Optional[_ImageItem] = None
        self.image_movable = False        # Move image mode
        self._outline: Optional[QGraphicsPolygonItem] = None
        self.node_items: dict[str, _NodeItem] = {}
        self.corner_items: list[_CornerItem] = []
        self.scene().selectionChanged.connect(self._on_selection)

    # -- building --------------------------------------------------------

    def set_layout(self, layout: lo.Layout, image: Optional[QPixmap]) -> None:
        self.model = layout
        self.rebuild(image)
        self._zoomed = False
        self.fit()

    def rebuild(self, image: Optional[QPixmap] = None, keep_image: bool = False) -> None:
        if keep_image:
            self.fold_image_offset()      # a Move-image drag isn't lost
        pixmap = (self._image_item.pixmap() if keep_image and self._image_item else image)
        self.updating = True
        # clear() deletes selected items one by one, emitting selectionChanged
        # with items whose Node belongs to the old model.
        self.scene().blockSignals(True)
        try:
            self.scene().clear()
            self._image_item = None
            if pixmap is not None and not pixmap.isNull() and self.model.floorplan:
                self._image_item = _ImageItem(self, pixmap)
                self._image_item.setTransformationMode(
                    Qt.TransformationMode.SmoothTransformation)
                self.place_image()
                self.scene().addItem(self._image_item)
            self._outline = QGraphicsPolygonItem()
            pen = QPen(QColor("#444444"), 2)
            pen.setCosmetic(True)
            self._outline.setPen(pen)
            self._outline.setBrush(QBrush(QColor(255, 255, 255, 70)))
            self._outline.setZValue(0)
            self.scene().addItem(self._outline)
            self.update_outline()
            self.corner_items = []
            for i, (x, y) in enumerate(self.model.corners):
                item = _CornerItem(self, i)
                item.setPos(x, y)
                self.scene().addItem(item)
                self.corner_items.append(item)
            self.node_items = {}
            for node in self.model.nodes:
                item = _NodeItem(self, node)
                item.setPos(node.x, node.y)
                self.scene().addItem(item)
                self.node_items[node.mac] = item
            self.set_image_movable(self.image_movable)
        finally:
            self.scene().blockSignals(False)
            self.updating = False
        self.selection.emit(None)
        x0, y0, x1, y1 = self.model.bbox()
        self.scene().setSceneRect(QRectF(x0 - 50, y0 - 50, x1 - x0 + 100, y1 - y0 + 100))

    def update_outline(self) -> None:
        if self._outline is not None:
            self._outline.setPolygon(QPolygonF([QPointF(x, y) for x, y in self.model.corners]))

    def place_image(self) -> None:
        """Map image pixels to meters (Floorplan's mapping). The view flips
        y, so the image transform flips it back to stay upright."""
        fp = self.model.floorplan
        if self._image_item is None or fp is None:
            return
        s = 1.0 / fp.px_per_m
        ox, oy = fp.origin_px
        self._image_item.setTransform(QTransform(s, 0, 0, -s, -ox * s, oy * s))
        self._image_item.setPos(0, 0)
        self._image_item.setZValue(40 if self.image_movable else -10)
        self._image_item.setOpacity(fp.opacity)

    def set_image_movable(self, on: bool) -> None:
        """Move image mode: only the plan moves (Qt would otherwise drag the
        selected boards along with it), and it sits on top to be grabbed."""
        on = on and self._image_item is not None
        self.image_movable = on
        if on:
            self.scene().clearSelection()
        for item in [*self.node_items.values(), *self.corner_items]:
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not on)
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, not on)
        item = self._image_item
        if item is not None:
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, on)
            item.setZValue(40 if on else -10)
            if on:
                item.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                item.unsetCursor()

    def fold_image_offset(self) -> bool:
        """Turn a Move-image drag (the item's offset, in meters) into the
        floor plan's origin. True if it moved."""
        item, fp = self._image_item, self.model.floorplan
        if item is None or fp is None:
            return False
        dx, dy = item.pos().x(), item.pos().y()
        if not (dx or dy):
            return False
        fp.origin_px = (fp.origin_px[0] - dx * fp.px_per_m,
                        fp.origin_px[1] + dy * fp.px_per_m)
        self.updating = True
        try:
            self.place_image()            # resets pos to 0 with the new origin
        finally:
            self.updating = False
        return True

    def sync_positions(self) -> None:
        """Move items to the model's coordinates (after a table edit)."""
        self.updating = True
        try:
            for node in self.model.nodes:
                item = self.node_items.get(node.mac)
                if item is not None:
                    item.setPos(node.x, node.y)
                    item.refresh()
            for item, (x, y) in zip(self.corner_items, self.model.corners):
                item.setPos(x, y)
            self.update_outline()
        finally:
            self.updating = False

    # -- view ----------------------------------------------------------------

    def fit(self) -> None:
        x0, y0, x1, y1 = self.model.bbox()
        pad = max(0.5, 0.1 * max(x1 - x0, y1 - y0))
        # Extra room on the right for the board labels.
        width = x1 - x0 + 2 * pad + 0.4 * max(x1 - x0, 2.0)
        self.fitInView(QRectF(x0 - pad, y0 - pad, width, y1 - y0 + 2 * pad),
                       Qt.AspectRatioMode.KeepAspectRatio)

    def refit(self) -> None:
        """The Fit button: fit, and keep fitting on resize."""
        self._zoomed = False
        self.fit()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._zoomed:
            self.fit()                   # the first fit ran before we had a size

    def to_view(self, x: float, y: float):
        """Viewport pixel position of a point in meters."""
        return self.mapFromScene(QPointF(x, y))

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.15 ** (event.angleDelta().y() / 120)
        self._zoomed = True
        self.scale(factor, factor)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.fillRect(rect, QColor("#f4f4f4"))
        for step, color in ((0.5, QColor("#e2e2e2")), (1.0, QColor("#c8c8c8"))):
            pen = QPen(color)
            pen.setCosmetic(True)
            painter.setPen(pen)
            x = math.floor(rect.left() / step) * step
            while x <= rect.right():
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                x += step
            y = math.floor(rect.top() / step) * step
            while y <= rect.bottom():
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
                y += step

    # -- mouse -----------------------------------------------------------------

    def pick_points(self, n: int) -> None:
        """The next ``n`` clicks report their position (picked) instead of
        selecting or dragging."""
        self.picking = n
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def stop_picking(self) -> None:
        self.picking = 0
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.viewport().unsetCursor()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.picking and event.button() == Qt.MouseButton.LeftButton:
            p = self.mapToScene(event.position().toPoint())
            self.picking -= 1
            if not self.picking:
                self.stop_picking()
            self.picked.emit(p.x(), p.y())
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        p = self.mapToScene(event.position().toPoint())
        self.hovered.emit(p.x(), p.y())
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        """Double-click a wall to add a corner there."""
        p = self.mapToScene(event.position().toPoint())
        corners = self.model.corners
        tol = 8 / max(1e-9, self.transform().m11())         # 8 px in meters
        for i in range(len(corners)):
            a, b = corners[i], corners[(i + 1) % len(corners)]
            if lo._dist_to_segment((p.x(), p.y()), a, b) <= tol:
                j = self.model.insert_corner(i)
                x, y = ((_snap(p.x()), _snap(p.y())) if self.snap else (p.x(), p.y()))
                self.model.move_corner(j, x, y)
                self.rebuild(keep_image=True)
                self.edited.emit()
                return
        super().mouseDoubleClickEvent(event)

    def _on_selection(self) -> None:
        nodes = [i.node for i in self.scene().selectedItems() if isinstance(i, _NodeItem)]
        self.selection.emit(nodes[0] if len(nodes) == 1 else None)

    def select_node(self, node: Optional[lo.Node]) -> None:
        self.scene().blockSignals(True)
        self.scene().clearSelection()
        if node is not None and node.mac in self.node_items:
            self.node_items[node.mac].setSelected(True)
        self.scene().blockSignals(False)

    def image_pixmap(self) -> Optional[QPixmap]:
        return self._image_item.pixmap() if self._image_item else None


# --------------------------------------------------------------------------
# The tab
# --------------------------------------------------------------------------

class RoomTab(QWidget):
    """Room layout editor for links.json."""

    dirty_changed = pyqtSignal(bool)

    BOARD_COLUMNS = ("Role", "Label", "MAC", "X (m)", "Y (m)")

    def __init__(self, registry: dv.Registry, default_path: Path,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.registry = registry
        self.path = Path(default_path)
        self.dirty = False
        self._mtime: Optional[float] = None
        self._filling = False             # tables being refilled from the model
        self._scale_points: list[tuple[float, float]] = []

        self.canvas = RoomCanvas()
        self.canvas.edited.connect(self._on_canvas_edit)
        self.canvas.selection.connect(self._on_canvas_selection)
        self.canvas.hovered.connect(
            lambda x, y: self.coords.setText(f"x {x:5.2f} m   y {y:5.2f} m"))
        self.canvas.picked.connect(self._on_picked)
        self.canvas.image_moved.connect(lambda: self._set_dirty(True))

        # File row
        self.path_edit = QLineEdit(str(self.path))
        self.path_edit.setAccessibleName("links.json path")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.clicked.connect(lambda: self.load(Path(self.path_edit.text().strip())))
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(lambda: self.save())
        self.save_btn.setEnabled(False)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("links.json:"))
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(browse)
        file_row.addWidget(self.reload_btn)
        file_row.addWidget(self.save_btn)

        # View row
        self.snap_box = QCheckBox(f"Snap to {SNAP_M:g} m")
        self.snap_box.setChecked(True)
        self.snap_box.toggled.connect(lambda on: setattr(self.canvas, "snap", on))
        fit = QPushButton("Fit")
        fit.clicked.connect(self.canvas.refit)
        self.coords = QLabel("")
        self.coords.setFont(QFont("monospace"))
        view_row = QHBoxLayout()
        view_row.addWidget(self.snap_box)
        view_row.addWidget(fit)
        view_row.addStretch(1)
        view_row.addWidget(self.coords)

        # Boards
        self.boards = QTableWidget(0, len(self.BOARD_COLUMNS))
        self.boards.setAccessibleName("Boards")
        self.boards.setHorizontalHeaderLabels(self.BOARD_COLUMNS)
        self.boards.verticalHeader().setVisible(False)
        self.boards.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.boards.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.boards.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.boards.horizontalHeader().setStretchLastSection(True)
        self.boards.itemChanged.connect(self._on_board_cell)
        self.boards.itemSelectionChanged.connect(self._on_board_row)
        self.add_combo = QComboBox()
        self.add_combo.setAccessibleName("Board to add")
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self._add_board)
        self.remove_btn = QPushButton("Remove board")
        self.remove_btn.clicked.connect(self._remove_board)
        add_row = QHBoxLayout()
        add_row.addWidget(self.add_combo, 1)
        add_row.addWidget(self.add_btn)
        add_row.addWidget(self.remove_btn)

        # Room corners
        self.corners = QTableWidget(0, 2)
        self.corners.setAccessibleName("Corners")
        self.corners.setHorizontalHeaderLabels(("X (m)", "Y (m)"))
        self.corners.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.corners.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.corners.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.corners.itemChanged.connect(self._on_corner_cell)
        add_corner = QPushButton("Add corner")
        add_corner.setToolTip("Split the wall after the selected corner (or "
                              "double-click a wall on the plan).")
        add_corner.clicked.connect(self._add_corner)
        remove_corner = QPushButton("Remove corner")
        remove_corner.clicked.connect(self._remove_corner)
        corner_row = QHBoxLayout()
        corner_row.addWidget(add_corner)
        corner_row.addWidget(remove_corner)
        self.rect_w = QDoubleSpinBox()
        self.rect_h = QDoubleSpinBox()
        for box, name in ((self.rect_w, "Room width"), (self.rect_h, "Room depth")):
            box.setAccessibleName(name)
            box.setRange(0.5, 200.0)
            box.setDecimals(2)
            box.setSingleStep(0.1)
            box.setSuffix(" m")
        rect_btn = QPushButton("Make rectangle")
        rect_btn.clicked.connect(self._make_rectangle)
        rect_row = QHBoxLayout()
        rect_row.addWidget(self.rect_w)
        rect_row.addWidget(QLabel("×"))
        rect_row.addWidget(self.rect_h)
        rect_row.addWidget(rect_btn)

        # Floor plan image
        self.image_btn = QPushButton("Floor plan image…")
        self.image_btn.clicked.connect(self._choose_image)
        self.scale_btn = QPushButton("Set scale…")
        self.scale_btn.setToolTip("Click two points on the plan a known distance "
                                  "apart, then enter the distance.")
        self.scale_btn.clicked.connect(self._start_scale)
        self.move_image = QCheckBox("Move image")
        self.move_image.setToolTip("Drag the image to line it up with the room.")
        self.move_image.toggled.connect(self._toggle_move_image)
        self.remove_image_btn = QPushButton("Remove image")
        self.remove_image_btn.clicked.connect(self._remove_image)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setAccessibleName("Image opacity")
        self.opacity.setRange(5, 100)
        self.opacity.valueChanged.connect(self._on_opacity)
        image_row = QHBoxLayout()
        image_row.addWidget(self.image_btn)
        image_row.addWidget(self.scale_btn)
        image_row.addWidget(self.move_image)
        image_row.addWidget(self.remove_image_btn)
        self.scale_note = QLabel("")
        self.scale_note.setWordWrap(True)
        self.scale_dist = QDoubleSpinBox()
        self.scale_dist.setAccessibleName("Distance between the points")
        self.scale_dist.setRange(0.05, 500.0)
        self.scale_dist.setDecimals(2)
        self.scale_dist.setSuffix(" m")
        self.scale_apply = QPushButton("Apply scale")
        self.scale_apply.clicked.connect(self._apply_scale)
        self.scale_cancel = QPushButton("Cancel")
        self.scale_cancel.clicked.connect(self._cancel_scale)
        scale_row = QHBoxLayout()
        scale_row.addWidget(self.scale_dist)
        scale_row.addWidget(self.scale_apply)
        scale_row.addWidget(self.scale_cancel)
        self._scale_widgets = (self.scale_dist, self.scale_apply, self.scale_cancel)

        self.messages = QLabel("")
        self.messages.setWordWrap(True)

        side = QWidget()
        sv = QVBoxLayout(side)
        sv.addWidget(QLabel("<b>Boards</b> — drag them on the plan, or type x / y"))
        sv.addWidget(self.boards, 2)
        sv.addLayout(add_row)
        sv.addWidget(QLabel("<b>Room corners</b> — in order around the walls"))
        sv.addWidget(self.corners, 1)
        sv.addLayout(corner_row)
        sv.addLayout(rect_row)
        sv.addWidget(QLabel("<b>Floor plan</b>"))
        sv.addLayout(image_row)
        sv.addWidget(self.opacity)
        sv.addWidget(self.scale_note)
        sv.addLayout(scale_row)
        sv.addWidget(self.messages)

        split = QSplitter()
        split.addWidget(self.canvas)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.addLayout(file_row)
        lay.addLayout(view_row)
        lay.addWidget(split, 1)

        self._show_scale_row(False)
        self.load(self.path, quiet=True)

    def showEvent(self, event) -> None:  # noqa: N802
        # Boards flashed in the Devices tab or the wizard since last time.
        self._refresh_add_combo()
        super().showEvent(event)

    # -- file ----------------------------------------------------------------

    def load(self, path: Path, quiet: bool = False) -> bool:
        """Read links.json (a missing one starts an empty room)."""
        if self.dirty and not quiet and QMessageBox.question(
                self, "Discard changes?",
                "The room layout has unsaved changes. Discard them and reload?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return False
        path = Path(path).expanduser()
        try:
            layout = lo.Layout.load(path)
        except ValueError as exc:
            if not quiet:
                QMessageBox.warning(self, "Can't open links.json",
                                    f"{exc}\n\nFix the file, or choose another one.")
            self.messages.setText(f"⚠ Can't open {path}: {exc}")
            return False
        self.path = path
        self.path_edit.setText(str(path))
        self._mtime = path.stat().st_mtime if path.exists() else None
        image = None
        if layout.floorplan is not None:
            image = QPixmap(str(lo.resolve_image_path(layout.floorplan.image, path)))
            if image.isNull():
                image = None
        self.canvas.set_layout(layout, image)
        self._set_dirty(False)
        self._refresh_panel()
        if layout.floorplan is not None and image is None:
            self.messages.setText(f"⚠ Floor plan image not found: "
                                  f"{layout.floorplan.image}")
        return True

    def reload_if_clean(self, path: Optional[Path] = None) -> None:
        """After the wizard rewrote links.json (at ``path``): show its boards,
        unless that would throw away edits made here."""
        if not self.dirty:
            self.load(path or self.path, quiet=True)
        else:
            self.messages.setText("⚠ links.json was changed on disk (setup wizard). "
                                  "Save to keep your edits, or Reload to see its boards.")

    def save(self) -> bool:
        self.canvas.fold_image_offset()   # Move image may still be on
        layout = self.canvas.model
        problems = layout.problems()
        if problems:
            QMessageBox.warning(self, "Can't save yet", "\n\n".join(problems))
            return False
        path = Path(self.path_edit.text().strip()).expanduser()
        changed = path.exists() and (path != self.path or self._mtime is None
                                     or path.stat().st_mtime != self._mtime)
        if changed and QMessageBox.question(
                self, "Overwrite links.json?",
                f"{path} has changed since it was opened here (the setup wizard "
                f"may have added boards). Overwrite it with this layout?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return False
        fp = layout.floorplan
        if fp is not None and path.resolve() != self.path.resolve():
            # Saved elsewhere: the image path is relative to links.json.
            fp.image = lo.relative_image_path(
                str(lo.resolve_image_path(fp.image, self.path)), path)
        try:
            layout.save(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Couldn't save", f"{type(exc).__name__}: {exc}")
            return False
        self.path = path
        self._mtime = path.stat().st_mtime
        self._set_dirty(False)
        self._refresh_messages(saved=True)
        return True

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open links.json", str(self.path.parent),
                                              "links.json (*.json);;All files (*)")
        if path:
            self.load(Path(path))

    def _set_dirty(self, dirty: bool) -> None:
        if dirty != self.dirty:
            self.dirty = dirty
            self.dirty_changed.emit(dirty)
        self.save_btn.setEnabled(dirty)

    # -- panel ------------------------------------------------------------------

    def _refresh_panel(self) -> None:
        layout = self.canvas.model
        self._filling = True
        try:
            self.boards.setRowCount(len(layout.nodes))
            for row, n in enumerate(layout.nodes):
                cells = (n.role.upper(), n.label, n.mac, f"{n.x:.2f}", f"{n.y:.2f}")
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if col in (0, 2):
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if col == 1 and not n.placed:
                        item.setToolTip("Not placed yet: drag it to where the board is.")
                        item.setForeground(QBrush(QColor("#a06000")))
                    self.boards.setItem(row, col, item)
            self.corners.setRowCount(len(layout.corners))
            for row, (x, y) in enumerate(layout.corners):
                self.corners.setItem(row, 0, QTableWidgetItem(f"{x:.2f}"))
                self.corners.setItem(row, 1, QTableWidgetItem(f"{y:.2f}"))
            x0, y0, x1, y1 = (min(c[0] for c in layout.corners), min(c[1] for c in layout.corners),
                              max(c[0] for c in layout.corners), max(c[1] for c in layout.corners))
            self.rect_w.setValue(max(0.5, x1 - x0))
            self.rect_h.setValue(max(0.5, y1 - y0))
            self._refresh_add_combo()
            fp = layout.floorplan
            has_image = self.canvas.image_pixmap() is not None
            self.scale_btn.setEnabled(has_image)
            if not has_image:
                self.move_image.setChecked(False)
            self.move_image.setEnabled(has_image)
            self.remove_image_btn.setEnabled(fp is not None)
            self.opacity.setEnabled(has_image)
            if fp is not None:
                self.opacity.setValue(int(fp.opacity * 100))
        finally:
            self._filling = False
        self._refresh_messages()

    def _refresh_add_combo(self) -> None:
        layout = self.canvas.model
        self.add_combo.clear()
        for rec in self.registry.all():
            if rec.role in (dv.ROLE_TX, dv.ROLE_RX) and layout.find(rec.mac) is None:
                self.add_combo.addItem(f"{rec.role.upper()}  {rec.label or rec.mac}  "
                                       f"({rec.mac})", rec.mac)
        none = self.add_combo.count() == 0
        if none:
            self.add_combo.setPlaceholderText("No other flashed boards to add")
        else:
            # With placeholder text set, a refilled combo stays on "no
            # selection" and Add would silently do nothing.
            self.add_combo.setCurrentIndex(0)
        self.add_combo.setEnabled(not none)
        self.add_btn.setEnabled(not none)

    def _refresh_messages(self, saved: bool = False) -> None:
        layout = self.canvas.model
        lines = [f"⚠ {p}" for p in layout.problems()] + \
                [f"• {w}" for w in layout.warnings()]
        if saved:
            lines.insert(0, f"✓ Saved {self.path}")
        self.messages.setText("\n".join(lines))

    def _edited(self) -> None:
        self._set_dirty(True)
        self._refresh_panel()

    # -- canvas → model → panel -------------------------------------------------

    def _on_canvas_edit(self) -> None:
        self._edited()

    def _on_canvas_selection(self, node: Optional[lo.Node]) -> None:
        nodes = self.canvas.model.nodes
        row = next((i for i, n in enumerate(nodes) if n is node), None)
        if row is None:
            return
        self.boards.blockSignals(True)
        self.boards.selectRow(row)
        self.boards.blockSignals(False)

    def _selected_node(self) -> Optional[lo.Node]:
        rows = self.boards.selectionModel().selectedRows()
        if not rows:
            return None
        i = rows[0].row()
        nodes = self.canvas.model.nodes
        return nodes[i] if 0 <= i < len(nodes) else None

    def _on_board_row(self) -> None:
        self.canvas.select_node(self._selected_node())

    def _on_board_cell(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        layout = self.canvas.model
        node = layout.nodes[item.row()]
        col = item.column()
        if col == 1:
            node.label = item.text().strip()
        elif col in (3, 4):
            try:
                v = float(item.text().replace(",", "."))
                if not math.isfinite(v):
                    raise ValueError
            except ValueError:
                self._refresh_panel()            # put the old value back
                return
            x, y = (v, node.y) if col == 3 else (node.x, v)
            layout.move_node(node, x, y)
        self.canvas.sync_positions()
        self._edited()

    def _on_corner_cell(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        layout = self.canvas.model
        x, y = layout.corners[item.row()]
        try:
            v = float(item.text().replace(",", "."))
            if not math.isfinite(v):
                raise ValueError
        except ValueError:
            self._refresh_panel()
            return
        layout.move_corner(item.row(), *((v, y) if item.column() == 0 else (x, v)))
        self.canvas.sync_positions()
        self._edited()

    def _add_board(self) -> None:
        mac = self.add_combo.currentData()
        rec = self.registry.get(mac) if mac else None
        if rec is None:
            return
        node = self.canvas.model.add_node(rec.mac, rec.role, rec.label)
        self.canvas.rebuild(keep_image=True)
        self._edited()
        self.boards.selectRow(self.canvas.model.nodes.index(node))

    def _remove_board(self) -> None:
        node = self._selected_node()
        if node is None:
            return
        self.canvas.model.remove_node(node)
        self.canvas.rebuild(keep_image=True)
        self._edited()

    def _selected_corner(self) -> int:
        on_canvas = [i.index for i in self.canvas.scene().selectedItems()
                     if isinstance(i, _CornerItem)]
        if len(on_canvas) == 1:
            return on_canvas[0]
        rows = self.corners.selectionModel().selectedRows()
        return rows[0].row() if rows else len(self.canvas.model.corners) - 1

    def _add_corner(self) -> None:
        i = self.canvas.model.insert_corner(self._selected_corner())
        self.canvas.rebuild(keep_image=True)
        self._edited()
        self.corners.selectRow(i)

    def _remove_corner(self) -> None:
        if not self.canvas.model.remove_corner(self._selected_corner()):
            self.messages.setText(f"⚠ A room needs at least {lo.MIN_CORNERS} corners.")
            return
        self.canvas.rebuild(keep_image=True)
        self._edited()

    def _make_rectangle(self) -> None:
        self.canvas.model.set_rectangle(self.rect_w.value(), self.rect_h.value())
        self.canvas.rebuild(keep_image=True)
        self.canvas.refit()
        self._edited()

    # -- floor plan image ----------------------------------------------------------

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Floor plan image", str(self.path.parent),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.svg);;All files (*)")
        if path:
            self.set_image(Path(path))

    def set_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Can't open image",
                                f"{path} isn't an image Qt can read.")
            return False
        layout = self.canvas.model
        # Start with the image spanning the room's width, bottom-left corner
        # at the room's; then Set scale / Move image to line it up.
        xs = [c[0] for c in layout.corners]
        ys = [c[1] for c in layout.corners]
        ppm = pixmap.width() / max(0.5, max(xs) - min(xs))
        origin = (-min(xs) * ppm, pixmap.height() + min(ys) * ppm)
        layout.floorplan = lo.Floorplan(lo.relative_image_path(str(path), self.path),
                                        ppm, origin)
        self.canvas.rebuild(pixmap)
        self._edited()
        self.scale_note.setText("Now click Set scale… and click two points on the plan "
                                "whose real distance you know (e.g. the ends of a wall).")
        return True

    def _remove_image(self) -> None:
        self.canvas.model.floorplan = None
        self.canvas.rebuild()
        self._edited()

    def _on_opacity(self, value: int) -> None:
        fp = self.canvas.model.floorplan
        if self._filling or fp is None:
            return
        fp.opacity = value / 100
        self.canvas.place_image()
        self._set_dirty(True)

    def _toggle_move_image(self, on: bool) -> None:
        if not on:
            self.canvas.fold_image_offset()
        self.canvas.set_image_movable(on)
        self.scale_note.setText("Drag the image into place, then untick Move image."
                                if on else "")

    def _show_scale_row(self, show: bool) -> None:
        for w in self._scale_widgets:
            w.setVisible(show)

    def _start_scale(self) -> None:
        self._scale_points = []
        self.move_image.setChecked(False)
        self.canvas.pick_points(2)
        self.scale_note.setText("Click the first of two points a known distance apart.")

    def _on_picked(self, x: float, y: float) -> None:
        self._scale_points.append((x, y))
        if len(self._scale_points) == 1:
            self.scale_note.setText("Now click the second point.")
            return
        (ax, ay), (bx, by) = self._scale_points
        span = math.hypot(bx - ax, by - ay)
        if span < 1e-6:
            self.scale_note.setText("Those are the same point; click Set scale… again.")
            return
        self.scale_dist.setValue(round(span, 2))
        self.scale_note.setText(f"The points are {span:.2f} m apart at the current "
                                f"scale. Enter the real distance:")
        self._show_scale_row(True)

    def _apply_scale(self) -> None:
        fp = self.canvas.model.floorplan
        if fp is None or len(self._scale_points) != 2:
            return
        (ax, ay), (bx, by) = self._scale_points
        span_px = math.hypot(bx - ax, by - ay) * fp.px_per_m
        anchor = fp.to_pixels(ax, ay)            # this image pixel stays put
        fp.px_per_m = span_px / self.scale_dist.value()
        fp.origin_px = (anchor[0] - ax * fp.px_per_m, anchor[1] + ay * fp.px_per_m)
        self.canvas.place_image()
        self._cancel_scale(done=True)
        self._set_dirty(True)

    def _cancel_scale(self, done: bool = False) -> None:
        self.canvas.stop_picking()
        self._scale_points = []
        self._show_scale_row(False)
        self.scale_note.setText("Scale set. Tick Move image to line the plan up with the "
                                "room." if done else "")
