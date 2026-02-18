import os
import shutil
import subprocess
import sys
import tempfile
import wave
import math
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QSlider,
    QSizePolicy,
    QProgressBar,
    QSizeGrip,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

DEFAULT_RATE = 180
DEFAULT_PITCH = 1.0
AUTO_GENERATE_DELAY_MS = 500
AUTO_GENERATE_DELAY_LARGE_MS = 1200
LARGE_TEXT_THRESHOLD = 800
PASTE_DELTA_THRESHOLD = 200
MIN_LENGTH_SCALE = 0.5
MAX_LENGTH_SCALE = 2.0
PITCH_MIN = 0.5
PITCH_MAX = 2.0


@dataclass
class TTSConfig:
    voice_id: str
    rate: int
    volume: float
    pitch: float


class DraggableHeader(QWidget):
    def __init__(self, parent: QWidget, title: str, action_buttons: list[QWidget]):
        super().__init__(parent)
        self._window = parent
        self._drag_pos = None

        self.setObjectName("headerBar")
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 8, 6)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("headerTitle")
        layout.addWidget(self.title_label)

        layout.addStretch(1)
        for btn in action_buttons:
            btn.setMinimumHeight(28)
            layout.addWidget(btn)
        layout.addSpacing(8)

        self.min_btn = QPushButton("_")
        self.min_btn.setObjectName("minButton")
        self.min_btn.setFixedSize(52, 40)

        self.close_btn = QPushButton("X")
        self.close_btn.setObjectName("closeButton")
        self.close_btn.setFixedSize(60, 40)

        layout.addWidget(self.min_btn)
        layout.addWidget(self.close_btn)

        self.min_btn.clicked.connect(self._window.showMinimized)
        self.close_btn.clicked.connect(self._window.close)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self._window.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self._window.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._window.isMaximized():
                self._window.showNormal()
            else:
                self._window.showMaximized()
            event.accept()


class TTSEngineService:
    def __init__(self, voices_dir: Path):
        self._voices_dir = voices_dir

    def set_voices_dir(self, voices_dir: Path):
        self._voices_dir = voices_dir

    def list_voices(self):
        voices = []
        for onnx_path in sorted(self._voices_dir.glob("*.onnx")):
            voices.append((str(onnx_path), onnx_path.stem))
        return voices

    def _ensure_model_config(self, model_path: Path) -> Path | None:
        expected = Path(str(model_path) + ".json")
        if expected.exists():
            return expected

        alt = model_path.with_suffix(".json")
        if alt.exists():
            try:
                shutil.copyfile(alt, expected)
                return expected
            except Exception:
                return None
        return None

    def _check_dependencies(self):
        try:
            import importlib.util
        except Exception as exc:
            raise RuntimeError("Falha ao verificar dependencias do Piper.") from exc

        if importlib.util.find_spec("piper") is None:
            raise RuntimeError(
                "piper-tts nao instalado. Execute: py -3 -m pip install piper-tts"
            )
        if importlib.util.find_spec("pathvalidate") is None:
            raise RuntimeError(
                "Dependencia 'pathvalidate' ausente. Execute: py -3 -m pip install pathvalidate"
            )

    def _rate_to_length_scale(self, rate: int) -> float:
        if rate <= 0:
            return 1.0
        scale = DEFAULT_RATE / rate
        return max(MIN_LENGTH_SCALE, min(MAX_LENGTH_SCALE, scale))

    def synthesize_to_wav(self, text: str, config: TTSConfig, out_path: str):
        self._check_dependencies()

        model_path = Path(config.voice_id)
        if not model_path.exists():
            raise RuntimeError("Modelo .onnx nao encontrado.")

        if not self._ensure_model_config(model_path):
            raise RuntimeError(
                "Arquivo .onnx.json nao encontrado. Renomeie o .json para .onnx.json."
            )

        length_scale = self._rate_to_length_scale(config.rate)
        cmd = [
            sys.executable,
            "-m",
            "piper",
            "-m",
            str(model_path),
            "-f",
            out_path,
            "--length_scale",
            f"{length_scale:.3f}",
            "--",
            text,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(detail or "Falha ao executar o Piper.")

        # Pitch e volume serao aplicados em pos-processamento (preview/salvar)

    def apply_pitch_to_wav(self, src_path: str, dest_path: str, pitch: float):
        if abs(pitch - 1.0) < 0.001:
            if src_path != dest_path:
                shutil.copyfile(src_path, dest_path)
            return

        if pitch < PITCH_MIN or pitch > PITCH_MAX:
            raise RuntimeError("Pitch fora do limite permitido (50% a 200%).")

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError(
                "Para ajustar o pitch, instale o ffmpeg ou mantenha o pitch em 100."
            )

        with wave.open(src_path, "rb") as wf:
            framerate = wf.getframerate()

        asetrate = max(8000, int(framerate * pitch))
        atempo = 1.0 / pitch
        filter_arg = f"asetrate={asetrate},atempo={atempo:.5f}"

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cmd = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            src_path,
            "-filter:a",
            filter_arg,
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(detail or "Falha ao ajustar o pitch.")

        if dest_path == src_path:
            shutil.move(tmp_path, dest_path)
        else:
            shutil.move(tmp_path, dest_path)

    def _apply_volume_to_wav(self, src_path: str, dest_path: str, volume: float):
        volume_factor = max(0.0, min(2.0, volume))
        if volume_factor == 1.0:
            if src_path != dest_path:
                shutil.copyfile(src_path, dest_path)
            return

        with wave.open(src_path, "rb") as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())

        sampwidth = params.sampwidth
        nchannels = params.nchannels
        framerate = params.framerate

        if sampwidth != 2:
            return

        import array

        samples = array.array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()

        max_amp = 32767
        min_amp = -32768
        for i, value in enumerate(samples):
            scaled = int(value * volume_factor)
            if scaled > max_amp:
                scaled = max_amp
            elif scaled < min_amp:
                scaled = min_amp
            samples[i] = scaled

        if sys.byteorder != "little":
            samples.byteswap()
        frames = samples.tobytes()

        with wave.open(dest_path, "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(framerate)
            wf.writeframes(frames)

    def convert_wav_to_mp3(self, wav_path: str, mp3_path: str, volume: float = 1.0):
        try:
            from pydub import AudioSegment
        except Exception as exc:
            raise RuntimeError(
                "pydub nao instalado. Instale pydub e ffmpeg para exportar mp3."
            ) from exc

        audio = AudioSegment.from_wav(wav_path)
        if volume < 0.001:
            audio = audio - 120
        elif abs(volume - 1.0) >= 0.001:
            gain_db = 20.0 * math.log10(volume)
            audio = audio.apply_gain(gain_db)
        audio.export(mp3_path, format="mp3")


class SynthesisWorker(QThread):
    finished = pyqtSignal(bool, str, str)

    def __init__(self, service: TTSEngineService, text: str, config: TTSConfig, out_path: str):
        super().__init__()
        self._service = service
        self._text = text
        self._config = config
        self._out_path = out_path

    def run(self):
        try:
            self._service.synthesize_to_wav(self._text, self._config, self._out_path)
            self.finished.emit(True, "OK", self._out_path)
        except Exception as exc:
            self.finished.emit(False, str(exc), "")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text to Audio - TTS Studio")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setMinimumSize(720, 560)

        self._voices_dir = Path(__file__).resolve().parent
        self._assets_dir = self._voices_dir
        self._service = TTSEngineService(self._voices_dir)
        self._last_wav_path = ""
        self._preview_wav_path = ""
        self._preview_pitch = 1.0
        self._worker = None
        self._pending_preview = False
        self._pending_save_path = ""
        self._auto_generate_enabled = False
        self._auto_pending = False
        self._is_busy = False
        self._seeking = False
        self._last_text_len = 0

        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(AUTO_GENERATE_DELAY_MS)
        self._auto_timer.timeout.connect(self._auto_generate)

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._duration_ms = 0

        self._build_ui()
        self._apply_icons()
        self._on_volume_changed()
        self._load_voices(show_warning=False)
        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #485161;
                color: #e6e6e6;
                font-size: 12px;
            }
            #headerBar {
                background-color: #051F1C;
            }
            QLabel {
                color: #f2f2f2;
            }

            #headerTitle {
                font-weight: 600;
                font-size: 13px;
                color: #FFF;
            }
            #headerBar QPushButton {
                color: #FFF;
            }
            QPushButton {
                background-color: #323232;
                border: 1px solid #3a3a3a;
                padding: 4px 10px;
                border-radius: 6px;
            }
            #headerBar QPushButton {
                background-color: transparent;
                border: none;
            }
            #clearButton {
                background-color: transparent;
                border: none;
            }
            QPushButton:hover {
                background-color: #3b3b3b;
            }
            QPushButton:pressed {
                background-color: #2b2b2b;
            }
            #closeButton {
                background-color: #8a2f2f;
                border-color: #9a3b3b;
                font-weight: 600;
            }
            #closeButton:hover {
                background-color: #b64545;
            }
            #stopButton {
                background-color: #3a3a3a;
                border-color: #4a4a4a;
                font-weight: 600;
            }
            #stopButton:hover {
                background-color: #4a4a4a;
            }
            #minButton {
                font-weight: 700;
            }
            QComboBox, QSpinBox, QTextEdit {
                background-color: #100626;
                border: 1px solid #fff;
                font-weight: 500;
                border-radius: 6px;
            }
            QComboBox {
                min-height: 20px;
                padding: 2px 8px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 28px;
                padding: 6px 8px;
            }
            QTextEdit {
                padding: 6px;
            }
            QLabel#voiceDirLabel {
                color: #bdbdbd;
            }
            QScrollBar:vertical {
                background: #1a1a1a;
                width: 12px;
                margin: 2px 2px 2px 2px;
                border-radius: 6px;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: #1a1a1a;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #23d0a6;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #2ce2b5;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #1a1a1a;
                height: 12px;
                margin: 2px 2px 2px 2px;
                border-radius: 6px;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: #1a1a1a;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #23d0a6;
                min-width: 24px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #2ce2b5;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 6px;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #23d0a6;
                border-radius: 6px;
            }
            #playbackSlider::groove:horizontal {
                background: #1a1a1a;
                border: 1px solid #333;
                height: 8px;
                border-radius: 4px;
            }
            #playbackSlider::sub-page:horizontal {
                background: #23d0a6;
                border-radius: 4px;
            }
            #playbackSlider::add-page:horizontal {
                background: #1a1a1a;
                border-radius: 4px;
            }
            #playbackSlider::handle:horizontal {
                background: #f2f2f2;
                border: 2px solid #0e6f5a;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            #playbackSlider::handle:horizontal:hover {
                background: #ffffff;
            }
            """
        )

    def _icon_path(self, filename: str) -> Path:
        return self._assets_dir / filename

    def _load_icon(self, filename: str) -> QIcon | None:
        path = self._icon_path(filename)
        if path.exists():
            return QIcon(str(path))
        return None

    def _set_button_icon(self, button: QPushButton, filename: str, tooltip: str, size: int):
        icon = self._load_icon(filename)
        button.setToolTip(tooltip)
        if icon is None:
            button.setText(tooltip)
            return
        button.setIcon(icon)
        button.setIconSize(QSize(size, size))
        button.setText("")

    def _set_label_icon(self, label: QLabel, filename: str, tooltip: str, size: int):
        label.setToolTip(tooltip)
        path = self._icon_path(filename)
        if not path.exists():
            label.setText(tooltip)
            return
        pixmap = QPixmap(str(path)).scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)
        label.setFixedSize(size + 6, size + 6)
        label.setText("")

    def _update_preview_icon(self, state):
        if self._play_icon is None or self._pause_icon is None:
            return
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.preview_btn.setIcon(self._pause_icon)
            self.preview_btn.setIconSize(QSize(20, 20))
        else:
            self.preview_btn.setIcon(self._play_icon)
            self.preview_btn.setIconSize(QSize(20, 20))
        self.preview_btn.setText("")

    def _apply_icons(self):
        self._play_icon = self._load_icon("reproduzir.png")
        self._pause_icon = self._load_icon("pausa.png")

        self._set_button_icon(self.generate_btn, "gerar.png", "Gerar audio", 20)
        self._set_button_icon(self.stop_btn, "stop.png", "Stop", 20)
        self._set_button_icon(self.save_btn, "save.png", "Salvar", 20)
        self._set_button_icon(self.clear_btn, "limpar.png", "Limpar texto", 18)
        self._set_button_icon(self.voice_dir_btn, "folder.png", "Escolher pasta", 18)

        if self._play_icon is not None:
            self.preview_btn.setIcon(self._play_icon)
            self.preview_btn.setIconSize(QSize(20, 20))
            self.preview_btn.setText("")
            self.preview_btn.setToolTip("Reproduzir / Pausar")
        else:
            self.preview_btn.setToolTip("Reproduzir / Pausar")

        self._set_button_icon(self.header.min_btn, "mini.png", "Minimizar", 24)
        self._set_button_icon(self.header.close_btn, "fechar.png", "Fechar", 24)

        self._set_label_icon(self.volume_label, "volume.png", "Volume", 16)
        self._set_label_icon(self.pitch_label, "pitch.png", "Pitch", 16)

        header_size = QSize(60, 40)
        for btn in [self.generate_btn, self.preview_btn, self.stop_btn, self.save_btn]:
            btn.setFixedSize(header_size)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setFixedSize(32, 30)
        self.voice_dir_btn.setFixedSize(32, 30)
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.voice_dir_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.min_btn.setFixedSize(header_size)
        self.header.close_btn.setFixedSize(header_size)
        self.header.min_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.auto_toggle.setCursor(Qt.CursorShape.PointingHandCursor)

    def _build_ui(self):
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Digite ou cole o texto aqui...")
        self.text_edit.setMinimumHeight(220)
        self.text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.text_edit.textChanged.connect(self._on_text_changed)

        self.clear_btn = QPushButton("Limpar")
        self.clear_btn.setObjectName("clearButton")
        self.clear_btn.clicked.connect(self.text_edit.clear)

        self.voice_combo = QComboBox()
        self.voice_combo.setFixedHeight(36)
        self.voice_combo.currentIndexChanged.connect(self._on_params_changed)

        self.voice_dir_btn = QPushButton("Selecionar Pasta")
        self.voice_dir_btn.clicked.connect(self.on_change_voices_dir)

        self.voice_dir_label = QLabel(str(self._voices_dir))
        self.voice_dir_label.setObjectName("voiceDirLabel")
        self.voice_dir_label.setToolTip(str(self._voices_dir))

        self.rate_combo = QComboBox()
        rate_options = [
            ("Lenticiomo", 90),
            ("Muito Lento", 120),
            ("Lento", 150),
            ("Normal", 180),
            ("Rapido", 210),
            ("Muito Rapido", 240),
            ("Super Rapido", 300),
        ]
        for label, value in rate_options:
            self.rate_combo.addItem(f"{label} ({value})", value)
        default_index = next(
            (i for i, (_label, value) in enumerate(rate_options) if value == DEFAULT_RATE),
            0,
        )
        self.rate_combo.setCurrentIndex(default_index)
        self.rate_combo.setFixedHeight(36)
        self.rate_combo.currentIndexChanged.connect(self._on_params_changed)

        self.volume_combo = QComboBox()
        volume_options = [30, 50, 75, 90, 100]
        for value in volume_options:
            self.volume_combo.addItem(f"{value}%", value)
        default_volume_index = next(
            (i for i, value in enumerate(volume_options) if value == 90),
            0,
        )
        self.volume_combo.setCurrentIndex(default_volume_index)
        self.volume_combo.setFixedHeight(36)
        self.volume_combo.currentIndexChanged.connect(self._on_volume_changed)

        self.pitch_combo = QComboBox()
        pitch_options = [
            ("Grave", 50),
            ("Grave", 75),
            ("Normal", 100),
            ("Agudo", 125),
            ("Fina", 150),
            ("Mais Fina", 200),
        ]
        for label, value in pitch_options:
            self.pitch_combo.addItem(f"{label} ({value}%)", value)
        default_pitch_index = next(
            (i for i, (_label, value) in enumerate(pitch_options) if value == 100),
            0,
        )
        self.pitch_combo.setCurrentIndex(default_pitch_index)
        self.pitch_combo.setFixedHeight(36)
        self.pitch_combo.currentIndexChanged.connect(self._on_pitch_changed)

        self.volume_label = QLabel("Volume")
        self.pitch_label = QLabel("Pitch")

        self.generate_btn = QPushButton("Gerar Audio")
        self.preview_btn = QPushButton("Reproduzir Preview")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setFixedHeight(34)
        self.stop_btn.setFixedWidth(90)
        self.save_btn = QPushButton("Salvar (.mp3/.wav)")
        self.auto_toggle = QCheckBox("Auto-gerar")
        self.auto_toggle.setChecked(False)
        self.auto_toggle.toggled.connect(self._on_auto_toggle)
        self.auto_toggle.setToolTip("Ativar / desativar auto-geracao")

        self.generate_btn.clicked.connect(self.on_generate)
        self.preview_btn.clicked.connect(self.on_preview)
        self.stop_btn.clicked.connect(self.on_stop)
        self.save_btn.clicked.connect(self.on_save)

        self.header = DraggableHeader(
            self,
            "Text to Audio - TTS Studio",
            [self.auto_toggle, self.generate_btn, self.preview_btn, self.stop_btn, self.save_btn],
        )

        self.status_label = QLabel("Pronto")

        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.setObjectName("playbackSlider")
        self.playback_slider.setRange(0, 1000)
        self.playback_slider.setValue(0)
        self.playback_slider.setTracking(True)
        self.playback_slider.sliderPressed.connect(self._on_seek_start)
        self.playback_slider.sliderReleased.connect(self._on_seek_end)
        self.playback_slider.sliderMoved.connect(self._on_seek_move)

        self.generation_progress = QProgressBar()
        self.generation_progress.setTextVisible(False)
        self.generation_progress.setRange(0, 1)
        self.generation_progress.setValue(0)

        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice_combo, 1)
        voice_row.addWidget(self.voice_dir_btn)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(QLabel("Voz (.onnx)"))
        controls_layout.addLayout(voice_row)
        controls_layout.addWidget(self.voice_dir_label)
        controls_layout.addWidget(QLabel("Velocidade"))
        controls_layout.addWidget(self.rate_combo)
        volume_col = QVBoxLayout()
        volume_col.setContentsMargins(0, 0, 0, 0)
        volume_col.setSpacing(6)
        volume_col.addWidget(self.volume_label)
        volume_col.addWidget(self.volume_combo)

        pitch_col = QVBoxLayout()
        pitch_col.setContentsMargins(0, 0, 0, 0)
        pitch_col.setSpacing(6)
        pitch_col.addWidget(self.pitch_label)
        pitch_col.addWidget(self.pitch_combo)

        volume_pitch_row = QHBoxLayout()
        volume_pitch_row.addLayout(volume_col, 1)
        volume_pitch_row.addLayout(pitch_col, 1)
        controls_layout.addLayout(volume_pitch_row)

        playback_col = QVBoxLayout()
        playback_col.setContentsMargins(0, 0, 0, 0)
        playback_col.setSpacing(6)
        playback_col.addWidget(QLabel("Reprodução"))
        playback_col.addWidget(self.playback_slider)

        generation_col = QVBoxLayout()
        generation_col.setContentsMargins(0, 0, 0, 0)
        generation_col.setSpacing(6)
        generation_col.addWidget(QLabel("Geração"))
        generation_col.addWidget(self.generation_progress)

        progress_row = QHBoxLayout()
        progress_row.addLayout(playback_col, 1)
        progress_row.addLayout(generation_col, 1)
        controls_layout.addLayout(progress_row)

        text_layout = QVBoxLayout()
        text_layout.addWidget(self.text_edit)
        text_layout.addWidget(self.clear_btn)

        footer_layout = QHBoxLayout()
        footer_layout.addWidget(self.status_label)
        footer_layout.addStretch(1)
        footer_layout.addWidget(QSizeGrip(self))

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self.header)
        layout.addLayout(text_layout)
        layout.addLayout(controls_layout)
        layout.addLayout(footer_layout)

        self.setLayout(layout)

    def _load_voices(self, show_warning: bool):
        self.voice_combo.clear()
        voices = self._service.list_voices()
        if not voices:
            if show_warning:
                QMessageBox.warning(
                    self,
                    "Aviso",
                    "Nenhum arquivo .onnx encontrado na pasta selecionada.",
                )
            return

        for voice_path, name in voices:
            self.voice_combo.addItem(name, voice_path)

    def _current_config(self) -> TTSConfig:
        voice_id = self.voice_combo.currentData()
        rate_data = self.rate_combo.currentData()
        rate = int(rate_data) if rate_data is not None else DEFAULT_RATE
        volume_data = self.volume_combo.currentData()
        volume = (int(volume_data) if volume_data is not None else 90) / 100.0
        pitch = self._current_pitch()
        return TTSConfig(voice_id=voice_id, rate=rate, volume=volume, pitch=pitch)

    def _current_pitch(self) -> float:
        pitch_data = self.pitch_combo.currentData()
        return (int(pitch_data) if pitch_data is not None else 100) / 100.0

    def _ensure_preview_audio(self) -> str:
        if not self._last_wav_path or not os.path.exists(self._last_wav_path):
            return ""

        pitch = self._current_pitch()
        if abs(pitch - 1.0) < 0.001:
            self._preview_wav_path = ""
            self._preview_pitch = pitch
            return self._last_wav_path

        if (
            self._preview_wav_path
            and abs(self._preview_pitch - pitch) < 0.001
            and os.path.exists(self._preview_wav_path)
        ):
            return self._preview_wav_path

        tmp_path = self._new_temp_wav()
        try:
            self._service.apply_pitch_to_wav(self._last_wav_path, tmp_path, pitch)
        except Exception as exc:
            QMessageBox.critical(self, "Erro", str(exc))
            return ""

        self._preview_wav_path = tmp_path
        self._preview_pitch = pitch
        return tmp_path

    def _ensure_text(self, show_warning: bool) -> str:
        text = self.text_edit.toPlainText().strip()
        if not text:
            if show_warning:
                QMessageBox.warning(self, "Aviso", "Digite algum texto para gerar o audio.")
            return ""
        return text

    def _new_temp_wav(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        return path

    def _set_busy(self, busy: bool):
        self._is_busy = busy
        self.generate_btn.setEnabled(not busy)
        self.preview_btn.setEnabled(not busy)
        self.save_btn.setEnabled(not busy)
        self.status_label.setText("Processando..." if busy else "Pronto")
        if busy:
            self.generation_progress.setRange(0, 0)
        else:
            self.generation_progress.setRange(0, 1)
            self.generation_progress.setValue(0)

    def _schedule_auto_generate(self, delay_ms: int | None = None):
        if not self._auto_generate_enabled:
            return
        if self._is_busy:
            self._auto_pending = True
            return
        if delay_ms is not None:
            self._auto_timer.setInterval(delay_ms)
        else:
            self._auto_timer.setInterval(AUTO_GENERATE_DELAY_MS)
        self._auto_timer.start()

    def _auto_generate(self):
        self._generate_audio(manual=False)

    def _on_text_changed(self):
        text_len = len(self.text_edit.toPlainText())
        delta = text_len - self._last_text_len
        self._last_text_len = text_len

        if delta >= PASTE_DELTA_THRESHOLD or text_len >= LARGE_TEXT_THRESHOLD:
            self._schedule_auto_generate(AUTO_GENERATE_DELAY_LARGE_MS)
        else:
            self._schedule_auto_generate()

    def _on_params_changed(self, _value=None):
        self._schedule_auto_generate()

    def _on_volume_changed(self, _value=None):
        volume_data = self.volume_combo.currentData()
        volume = (int(volume_data) if volume_data is not None else 90) / 100.0
        self._audio_output.setVolume(volume)

    def _on_pitch_changed(self, _value=None):
        self._preview_wav_path = ""
        self._preview_pitch = self._current_pitch()
        if self._player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self._player.stop()
            preview_path = self._ensure_preview_audio()
            if preview_path:
                self._player.setSource(QUrl.fromLocalFile(preview_path))
                self._player.play()

    def _on_auto_toggle(self, checked: bool):
        self._auto_generate_enabled = checked

    def on_change_voices_dir(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            "Selecionar pasta de vozes",
            str(self._voices_dir),
        )
        if not selected:
            return

        self._voices_dir = Path(selected)
        self._service.set_voices_dir(self._voices_dir)
        self.voice_dir_label.setText(str(self._voices_dir))
        self.voice_dir_label.setToolTip(str(self._voices_dir))
        self._last_wav_path = ""
        self._preview_wav_path = ""
        self._load_voices(show_warning=True)

    def _generate_audio(self, manual: bool):
        if self._is_busy:
            if not manual:
                self._auto_pending = True
            return

        text = self._ensure_text(show_warning=manual)
        if not text:
            return

        if self.voice_combo.currentData() is None:
            if manual:
                QMessageBox.warning(self, "Aviso", "Selecione uma voz .onnx.")
            return

        if manual:
            self._auto_generate_enabled = True

        out_path = self._new_temp_wav()
        config = self._current_config()
        self._set_busy(True)

        self._worker = SynthesisWorker(self._service, text, config, out_path)
        self._worker.finished.connect(self._on_generated)
        self._worker.start()

    def on_generate(self):
        self.auto_toggle.setChecked(True)
        self._generate_audio(manual=True)

    def _on_generated(self, ok: bool, message: str, path: str):
        self._set_busy(False)
        if not ok:
            QMessageBox.critical(self, "Erro", message)
            return

        self._last_wav_path = path
        self._preview_wav_path = ""
        self._preview_pitch = 1.0
        self.status_label.setText("Audio gerado em buffer temporario")

        if self._pending_preview:
            self._pending_preview = False
            self.on_preview()
            return

        if self._pending_save_path:
            pending_path = self._pending_save_path
            self._pending_save_path = ""
            self._save_to_path(pending_path)
            return

        if self._auto_pending:
            self._auto_pending = False
            self._auto_timer.start()

    def on_preview(self):
        if not self._last_wav_path or not os.path.exists(self._last_wav_path):
            self._pending_preview = True
            self.on_generate()
            return

        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self.status_label.setText("Pausado")
            return

        if self._player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            self._player.play()
            self.status_label.setText("Reproduzindo preview")
            return

        preview_path = self._ensure_preview_audio()
        if not preview_path:
            return
        self._on_volume_changed()
        self._player.setSource(QUrl.fromLocalFile(preview_path))
        self._player.play()
        self.status_label.setText("Reproduzindo preview")

    def on_stop(self):
        self._player.stop()
        self._player.setPosition(0)
        self.playback_slider.setValue(0)
        self.status_label.setText("Parado")

    def _on_duration_changed(self, duration: int):
        self._duration_ms = duration
        if duration <= 0:
            self.playback_slider.setRange(0, 1)
            self.playback_slider.setValue(0)
        else:
            self.playback_slider.setRange(0, duration)

    def _on_position_changed(self, position: int):
        if self._duration_ms > 0 and not self._seeking:
            self.playback_slider.setValue(position)

    def _on_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState:
            if self.playback_slider.value() < self._duration_ms:
                self.playback_slider.setValue(0)
        if self._play_icon is not None and self._pause_icon is not None:
            self._update_preview_icon(state)
            return

        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.preview_btn.setText("Pausar Preview")
        else:
            self.preview_btn.setText("Reproduzir Preview")

    def _on_seek_start(self):
        self._seeking = True

    def _on_seek_move(self, value: int):
        if self._duration_ms > 0:
            self._player.setPosition(value)

    def _on_seek_end(self):
        self._seeking = False
        if self._duration_ms > 0:
            self._player.setPosition(self.playback_slider.value())

    def _save_to_path(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        volume = self._current_config().volume
        pitch = self._current_pitch()
        temp_pitch = ""
        try:
            src_path = self._last_wav_path
            if abs(pitch - 1.0) >= 0.001:
                temp_pitch = self._new_temp_wav()
                self._service.apply_pitch_to_wav(src_path, temp_pitch, pitch)
                src_path = temp_pitch
            if ext == ".mp3":
                self._service.convert_wav_to_mp3(src_path, file_path, volume)
            else:
                if abs(volume - 1.0) < 0.001:
                    shutil.copyfile(src_path, file_path)
                else:
                    self._service._apply_volume_to_wav(src_path, file_path, volume)
        except Exception as exc:
            QMessageBox.critical(self, "Erro", str(exc))
            return
        finally:
            if temp_pitch and os.path.exists(temp_pitch):
                try:
                    os.remove(temp_pitch)
                except Exception:
                    pass

        self.status_label.setText(f"Salvo: {file_path}")

    def on_save(self):
        text = self._ensure_text(show_warning=True)
        if not text:
            return

        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Salvar audio",
            "",
            "WAV (*.wav);;MP3 (*.mp3)",
        )
        if not file_path:
            return

        if not self._last_wav_path or not os.path.exists(self._last_wav_path):
            self._pending_save_path = file_path
            self.on_generate()
            return

        self._save_to_path(file_path)


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
