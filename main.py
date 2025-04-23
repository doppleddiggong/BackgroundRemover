import sys
import os
import warnings
import logging
import numpy as np
import torch

from pathlib import Path
from io import BytesIO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QWidget, QLabel, QFileDialog, QProgressBar,
                           QSpacerItem, QSizePolicy, QComboBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import QPixmap, QImage, QFont, QDragEnterEvent, QDropEvent
from PIL import Image
from transparent_background import Remover

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 경고 메시지 필터링
warnings.filterwarnings('ignore')
logging.getLogger("transparent_background").setLevel(logging.ERROR)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class BackgroundRemoveThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, input_path, mode='fast', threshold=0.5, postprocess='보통'):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.input_path = input_path
        self.mode = mode
        self.threshold = float(threshold)
        self.postprocess = postprocess
        self.logger.debug(f"BackgroundRemoveThread 초기화: mode={mode}, threshold={threshold}, postprocess={postprocess}")

    def run(self):
        try:
            self.progress.emit(10)
            self.status.emit("이미지 로딩 중...")
            self.logger.debug("이미지 로딩 시작")
            
            # 입력 이미지 로드
            input_image = Image.open(self.input_path)
            self.logger.debug(f"이미지 로드됨: size={input_image.size}, mode={input_image.mode}")
            
            if input_image.mode != 'RGB':
                self.logger.debug(f"이미지 모드 변환: {input_image.mode} -> RGB")
                input_image = input_image.convert('RGB')
            
            self.progress.emit(30)
            self.status.emit(f"{self.mode} 모드로 모델 초기화 중...")
            self.logger.debug("모델 초기화 시작")
            
            # CPU 모드로 강제 설정
            self.logger.debug("CPU 모드로 강제 설정")
            device = 'cpu'
            
            # Remover 초기화 - fast 모드 사용
            try:
                self.logger.debug(f"Remover 초기화 시작: mode={self.mode}, device={device}")
                remover = Remover(
                    mode=self.mode,
                    jit=False,  # JIT 컴파일 비활성화
                    device=device
                )
                self.logger.debug("Remover 초기화 완료")
                self.status.emit("모델 로드 완료")
            except Exception as e:
                self.logger.error(f"모델 로드 실패: {str(e)}", exc_info=True)
                error_msg = str(e)
                if "CUDA" in error_msg:
                    error_msg = "GPU 메모리 부족 또는 CUDA 오류가 발생했습니다. CPU 모드를 사용해주세요."
                elif "download" in error_msg.lower():
                    error_msg = "모델 다운로드 중 오류가 발생했습니다. 인터넷 연결을 확인해주세요."
                elif "permission" in error_msg.lower():
                    error_msg = "파일 접근 권한이 없습니다. 관리자 권한으로 실행해보세요."
                self.error.emit(f"모델 로드 실패: {error_msg}")
                return
            
            self.progress.emit(50)
            self.status.emit("배경 제거 처리 중...")
            self.logger.debug("배경 제거 처리 시작")
            
            # 배경 제거
            try:
                self.logger.debug(f"process 호출: threshold={self.threshold}")
                output_image = remover.process(
                    input_image,
                    type='rgba',
                    threshold=self.threshold
                )
                self.logger.debug("배경 제거 처리 완료")
                
                if output_image is None:
                    self.logger.error("배경 제거 결과가 None입니다")
                    self.error.emit("배경 제거 결과가 없습니다. 다른 모드를 시도해보세요.")
                    return
                    
            except Exception as e:
                self.logger.error(f"배경 제거 실패: {str(e)}", exc_info=True)
                self.error.emit(f"배경 제거 실패: {str(e)}")
                return
            
            self.progress.emit(80)
            self.status.emit("후처리 중...")
            self.logger.debug("후처리 시작")
            
            # 후처리 적용
            if isinstance(output_image, np.ndarray):
                self.logger.debug(f"출력 이미지 변환: ndarray -> PIL, shape={output_image.shape}")
                output_image = Image.fromarray(output_image)
            
            if output_image.mode != 'RGBA':
                self.logger.debug(f"이미지 모드 변환: {output_image.mode} -> RGBA")
                output_image = output_image.convert('RGBA')
            
            # 배경 제거 강도 조절
            if self.postprocess != '기본':
                try:
                    self.logger.debug(f"후처리 적용: {self.postprocess}")
                    r, g, b, a = output_image.split()
                    if self.postprocess == '더 많이 남기기':
                        self.logger.debug("알파 채널 20% 감소")
                        a = a.point(lambda x: max(int(x * 0.8), 0))
                    elif self.postprocess == '보통':
                        self.logger.debug("알파 채널 10% 증가")
                        a = a.point(lambda x: min(int(x * 1.1), 255))
                    elif self.postprocess == '더 많이 지우기':
                        self.logger.debug("알파 채널 강한 임계값 적용")
                        a = a.point(lambda x: 255 if x > 128 else max(int(x * 0.5), 0))
                    
                    # 결과 이미지 합성
                    output_image = Image.merge('RGBA', (r, g, b, a))
                    self.logger.debug("후처리 완료")
                        
                except Exception as e:
                    self.logger.error(f"알파 채널 처리 실패: {str(e)}", exc_info=True)
                    self.error.emit(f"알파 채널 처리 실패: {str(e)}")
                    return
            
            self.progress.emit(100)
            self.status.emit("처리 완료!")
            self.logger.debug("모든 처리 완료")
            self.finished.emit(output_image)
            
        except Exception as e:
            self.logger.error(f"처리 중 오류 발생: {str(e)}", exc_info=True)
            self.error.emit(f"처리 중 오류 발생: {str(e)}")

class DragDropLabel(QLabel):
    dropped = pyqtSignal(str)  # 드롭된 파일 경로를 전달하는 시그널

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("이미지를 여기로 드래그하세요")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 10px;
            }
            QLabel:hover {
                border-color: #4CAF50;
                background-color: #f0f0f0;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        QLabel {
                            border: 2px dashed #4CAF50;
                            border-radius: 5px;
                            background-color: #e8f5e9;
                            padding: 10px;
                        }
                    """)
                    return
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #f44336;
                border-radius: 5px;
                background-color: #ffebee;
                padding: 10px;
            }
        """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 10px;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.dropped.emit(file_path)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 10px;
            }
        """)

class BackgroundRemover(QMainWindow):
    def __init__(self):
        super().__init__()
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("배경 제거 도구")
        self.setGeometry(100, 100, 1200, 800)
        
        # 메인 위젯과 레이아웃 설정
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 옵션 컨테이너
        options_container = QWidget()
        options_layout = QHBoxLayout(options_container)
        options_layout.setSpacing(20)
        
        # 모드 선택
        mode_container = QWidget()
        mode_layout = QVBoxLayout(mode_container)
        mode_label = QLabel("모드:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['base', 'fast'])
        self.mode_combo.setCurrentText('base')
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        
        # 모드 설명 레이블 추가
        self.mode_desc_label = QLabel()
        self.mode_desc_label.setWordWrap(True)
        self.mode_desc_label.setStyleSheet("color: #666666; font-size: 11px;")
        mode_layout.addWidget(self.mode_desc_label)
        
        # 모드 변경 시 설명 업데이트
        self.mode_combo.currentTextChanged.connect(self.update_mode_description)
        self.update_mode_description(self.mode_combo.currentText())
        
        options_layout.addWidget(mode_container)
        
        # 임계값 선택
        threshold_container = QWidget()
        threshold_layout = QVBoxLayout(threshold_container)
        threshold_label = QLabel("임계값:")
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
        self.threshold_combo.setCurrentText('0.5')
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_combo)
        
        # 임계값 설명 레이블 추가
        self.threshold_desc_label = QLabel()
        self.threshold_desc_label.setWordWrap(True)
        self.threshold_desc_label.setStyleSheet("color: #666666; font-size: 11px;")
        threshold_layout.addWidget(self.threshold_desc_label)
        
        # 임계값 변경 시 설명 업데이트
        self.threshold_combo.currentTextChanged.connect(self.update_threshold_description)
        self.update_threshold_description(self.threshold_combo.currentText())
        
        options_layout.addWidget(threshold_container)
        
        # 후처리 선택
        postprocess_container = QWidget()
        postprocess_layout = QVBoxLayout(postprocess_container)
        postprocess_label = QLabel("배경 제거 강도:")
        self.postprocess_combo = QComboBox()
        self.postprocess_combo.addItems(['기본', '더 많이 남기기', '보통', '더 많이 지우기'])
        self.postprocess_combo.setCurrentText('보통')
        postprocess_layout.addWidget(postprocess_label)
        postprocess_layout.addWidget(self.postprocess_combo)
        
        # 후처리 설명 레이블 추가
        self.postprocess_desc_label = QLabel()
        self.postprocess_desc_label.setWordWrap(True)
        self.postprocess_desc_label.setStyleSheet("color: #666666; font-size: 11px;")
        postprocess_layout.addWidget(self.postprocess_desc_label)
        
        # 후처리 변경 시 설명 업데이트
        self.postprocess_combo.currentTextChanged.connect(self.update_postprocess_description)
        self.update_postprocess_description(self.postprocess_combo.currentText())
        
        options_layout.addWidget(postprocess_container)
        
        # 콤보박스 스타일
        combo_style = """
            QComboBox {
                padding: 5px;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                min-width: 120px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
        """
        self.mode_combo.setStyleSheet(combo_style)
        self.postprocess_combo.setStyleSheet(combo_style)
        
        layout.addWidget(options_container)
        
        # 버튼 컨테이너
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(20)
        
        # 버튼 스타일
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px;
                font-size: 14px;
                min-width: 150px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """
        
        # 이미지 선택 버튼
        self.select_btn = QPushButton("이미지 선택")
        self.select_btn.setStyleSheet(button_style)
        self.select_btn.clicked.connect(self.select_image)
        button_layout.addWidget(self.select_btn)
        
        # 저장 경로 선택 버튼
        self.save_path_btn = QPushButton("저장 경로 선택")
        self.save_path_btn.setStyleSheet(button_style)
        self.save_path_btn.clicked.connect(self.select_save_path)
        button_layout.addWidget(self.save_path_btn)
        
        # 배경 제거 버튼
        self.remove_bg_btn = QPushButton("배경 제거")
        self.remove_bg_btn.setStyleSheet(button_style)
        self.remove_bg_btn.clicked.connect(self.remove_background)
        self.remove_bg_btn.setEnabled(False)
        button_layout.addWidget(self.remove_bg_btn)
        
        layout.addWidget(button_container)
        
        # 진행 상태 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # 상태 레이블
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 이미지 표시 영역
        image_container = QWidget()
        image_layout = QHBoxLayout(image_container)
        image_layout.setSpacing(20)
        
        # 원본 이미지
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_title = QLabel("원본 이미지")
        original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        original_layout.addWidget(original_title)
        self.original_label = DragDropLabel()
        self.original_label.setMinimumSize(500, 500)
        self.original_label.dropped.connect(self.handle_dropped_image)
        original_layout.addWidget(self.original_label)
        image_layout.addWidget(original_container)
        
        # 결과 이미지
        result_container = QWidget()
        result_layout = QVBoxLayout(result_container)
        result_title = QLabel("결과 이미지")
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_layout.addWidget(result_title)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumSize(500, 500)
        self.result_label.setStyleSheet("border: 2px solid #cccccc; border-radius: 5px;")
        result_layout.addWidget(self.result_label)
        image_layout.addWidget(result_container)
        
        layout.addWidget(image_container)
        
        # 변수 초기화
        self.input_path = ""
        self.save_path = ""
        self.bg_thread = None

    def select_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "이미지 선택",
                "",
                "Images (*.png *.jpg *.jpeg)"
            )
            if file_name:
                self.logger.debug(f"선택된 파일: {file_name}")
                self.input_path = file_name
                self.display_image(file_name, self.original_label)
                self.remove_bg_btn.setEnabled(True)
                self.status_label.setText(f"선택된 이미지: {os.path.basename(file_name)}")
                self.result_label.clear()
                self.result_label.setText("결과가 여기에 표시됩니다")
        except Exception as e:
            self.logger.error(f"이미지 선택 중 오류: {str(e)}", exc_info=True)

    def select_save_path(self):
        try:
            save_path = QFileDialog.getExistingDirectory(self, "저장 경로 선택")
            if save_path:
                self.save_path = save_path
                self.status_label.setText(f"저장 경로: {save_path}")
        except Exception as e:
            self.logger.error(f"저장 경로 선택 중 오류: {str(e)}", exc_info=True)

    def create_checkerboard(self, size):
        """체커보드 패턴 배경 생성"""
        tile_size = 10
        width, height = size
        background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        draw = Image.new('RGBA', (tile_size * 2, tile_size * 2), (255, 255, 255, 0))
        
        # 체커보드 패턴 그리기
        for i in range(2):
            for j in range(2):
                if (i + j) % 2 == 0:
                    box = (i * tile_size, j * tile_size, 
                          (i + 1) * tile_size, (j + 1) * tile_size)
                    draw.paste((200, 200, 200, 255), box)
        
        # 패턴 반복
        for y in range(0, height, tile_size * 2):
            for x in range(0, width, tile_size * 2):
                background.paste(draw, (x, y))
        
        return background

    def display_image(self, image_path, label):
        if isinstance(image_path, str):
            # 파일에서 이미지 로드
            pixmap = QPixmap(image_path)
        else:  # PIL Image
            if image_path.mode == 'RGBA':
                # 체커보드 배경 생성
                background = self.create_checkerboard(image_path.size)
                # 이미지 합성
                background.paste(image_path, (0, 0), image_path)
                # PIL Image를 QPixmap으로 변환
                data = background.tobytes("raw", "RGBA")
                qim = QImage(data, background.size[0], background.size[1], QImage.Format.Format_RGBA8888)
            else:
                data = image_path.convert("RGBA").tobytes("raw", "RGBA")
                qim = QImage(data, image_path.size[0], image_path.size[1], QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qim)
        
        # 라벨 크기에 맞게 이미지 크기 조정
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def process_finished(self, output_image):
        try:
            if not self.save_path:
                self.save_path = os.path.dirname(self.input_path)
            
            # 결과 저장 - 파일명 중복 처리
            base_path = os.path.join(
                self.save_path, 
                f"nobg_{os.path.basename(self.input_path)}"
            )
            output_path = base_path
            counter = 1
            
            # 파일이 이미 존재하면 번호를 붙여서 새 파일명 생성
            while os.path.exists(output_path):
                name, ext = os.path.splitext(base_path)
                output_path = f"{name}_{counter}{ext}"
                counter += 1
            
            self.logger.debug(f"저장할 파일 경로: {output_path}")
            output_image.save(output_path)
            
            # 결과 이미지 표시
            self.display_image(output_image, self.result_label)
            self.status_label.setText(f"완료! 저장된 경로: {output_path}")
            
            # UI 상태 복원
            self.progress_bar.hide()
            self.remove_bg_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.save_path_btn.setEnabled(True)
        except Exception as e:
            self.logger.error(f"결과 처리 중 오류: {str(e)}", exc_info=True)
            self.status_label.setText(f"저장 중 오류 발생: {str(e)}")
            self.remove_bg_btn.setEnabled(True)
            self.select_btn.setEnabled(True)
            self.save_path_btn.setEnabled(True)

    def process_error(self, error_msg):
        self.status_label.setText(f"오류 발생: {error_msg}")
        self.progress_bar.hide()
        self.remove_bg_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.save_path_btn.setEnabled(True)

    def remove_background(self):
        if not self.input_path:
            self.status_label.setText("이미지를 먼저 선택해주세요")
            return
            
        # UI 상태 변경
        self.remove_bg_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.save_path_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # 백그라운드 처리 시작
        self.bg_thread = BackgroundRemoveThread(self.input_path, self.mode_combo.currentText(), self.threshold_combo.currentText(), self.postprocess_combo.currentText())
        self.bg_thread.progress.connect(self.update_progress)
        self.bg_thread.finished.connect(self.process_finished)
        self.bg_thread.error.connect(self.process_error)
        self.bg_thread.status.connect(self.status_label.setText)
        self.bg_thread.start()

    def update_mode_description(self, mode):
        descriptions = {
            'base': '기본 모드 - 대부분의 이미지에 적합한 균형잡힌 성능',
            'fast': '빠른 모드 - 처리 속도 우선',
            'simple': '단순 모드 - 간단한 배경 제거',
            'legacy': '레거시 모드 - 이전 버전 호환용'
        }
        self.mode_desc_label.setText(descriptions.get(mode, ''))

    def update_threshold_description(self, threshold):
        value = float(threshold)
        if value <= 0.3:
            desc = "낮은 임계값: 더 많은 부분을 전경으로 인식"
        elif value <= 0.6:
            desc = "중간 임계값: 균형잡힌 배경 제거"
        else:
            desc = "높은 임계값: 확실한 전경만 인식"
        self.threshold_desc_label.setText(desc)

    def update_postprocess_description(self, postprocess):
        descriptions = {
            '기본': '배경 제거 알고리즘의 기본 결과를 그대로 사용',
            '더 많이 남기기': '원본 이미지의 더 많은 부분을 보존 (흐릿한 가장자리 포함)',
            '보통': '기본값에서 배경을 약간 더 제거',
            '더 많이 지우기': '배경을 최대한 제거 (가장자리가 선명해짐)'
        }
        self.postprocess_desc_label.setText(descriptions.get(postprocess, ''))

    def handle_dropped_image(self, file_path):
        """드롭된 이미지 처리"""
        try:
            self.input_path = file_path
            self.display_image(file_path, self.original_label)
            self.remove_bg_btn.setEnabled(True)
            self.status_label.setText(f"선택된 이미지: {os.path.basename(file_path)}")
            self.result_label.clear()
            self.result_label.setText("결과가 여기에 표시됩니다")
        except Exception as e:
            self.logger.error(f"드롭된 이미지 처리 중 오류: {str(e)}", exc_info=True)
            self.status_label.setText("이미지 로드 중 오류가 발생했습니다")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일 설정
    app.setStyle("Fusion")
    
    window = BackgroundRemover()
    window.show()
    sys.exit(app.exec()) 