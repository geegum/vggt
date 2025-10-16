# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import threading

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# --- 설정 ---
FIXED_TARGET_DIR = "/home/etrlab/vggt"
WEBCAM_INDICES = [0, 2] 
# 웹캠 이미지를 재구성에 사용할 'images' 폴더에 바로 저장합니다.
SAVE_DIRECTORY = os.path.join(FIXED_TARGET_DIR, 'images')

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- [추가 1] 실시간 업데이트를 위한 전역 변수 및 Lock 객체 ---
# 새로운 3D 모델 파일(.glb)이 생성되었는지 공유하기 위한 변수들입니다.
LATEST_GLB_FILE = None
new_reconstruction_available = threading.Event() # 이벤트 객체로 상태 변경을 알림

# --- 모델 로딩 ---
print("Initializing and loading VGGT model...")
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()
model = model.to(device)

# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    'target_dir/images' 폴더의 이미지에 대해 VGGT 모델을 실행하고 예측 결과를 반환합니다.
    """
    print(f"Processing images from {target_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    model = model.to(device)
    model.eval()

    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    print(f"Found {len(image_names)} images")
    if len(image_names) < 2: # 최소 2개의 이미지가 필요
        print(f"Warning: At least 2 images are required for reconstruction. Found {len(image_names)}.")
        return None

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions['pose_enc_list'] = None

    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    torch.cuda.empty_cache()
    return predictions


# --- [수정 1] 재사용을 위해 재구성 로직을 별도 함수로 분리 ---
def perform_reconstruction(target_dir, conf_thres=50.0, frame_filter="All", mask_black_bg=False, mask_white_bg=False, show_cam=True, mask_sky=False, prediction_mode="Depthmap and Camera Branch"):
    """
    주어진 디렉토리의 이미지를 사용하여 3D 재구성을 수행하고 결과물(.glb 파일) 경로를 반환합니다.
    """
    global LATEST_GLB_FILE
    
    if not os.path.isdir(target_dir):
        print("Error: Target directory not found.")
        return None, "Error: Target directory not found.", []

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    print("Running model for reconstruction...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    if predictions is None:
        return None, "Reconstruction failed: Not enough images.", []

    # GLB 파일 이름 생성
    glbfile = os.path.join(target_dir, f"reconstruction_{int(time.time())}.glb")

    print("Converting predictions to GLB file...")
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    log_msg = f"Reconstruction Complete! ({end_time - start_time:.2f}s)"
    print(log_msg)

    # 전역 변수 업데이트 및 이벤트 설정
    LATEST_GLB_FILE = glbfile
    new_reconstruction_available.set() # UI에 업데이트가 필요함을 알림

    image_paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    return glbfile, log_msg, image_paths


# --- [수정 2] 웹캠 캡처 및 재구성 트리거 함수 ---
def opencv_capture_and_reconstruct():
    """
    웹캠을 제어하고, 스페이스바를 누르면 이미지를 캡처한 후
    `perform_reconstruction` 함수를 호출하여 재구성을 트리거합니다.
    """
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    print(f"이미지는 '{SAVE_DIRECTORY}' 경로에 저장됩니다.")

    caps = [cv2.VideoCapture(i) for i in WEBCAM_INDICES]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"오류: 웹캠 {WEBCAM_INDICES[i]}번을 열 수 없습니다.")
            exit()
        print(f"웹캠 {WEBCAM_INDICES[i]}번이 성공적으로 열렸습니다.")
    
    print("\n웹캠 화면이 보이면 스페이스바를 눌러 실시간 3D 재구성을 시작하세요.")
    print("'q' 키를 누르면 프로그램이 종료됩니다.")

    while True:
        frames = [cap.read() for cap in caps]
        valid_frames = [frame for ret, frame in frames if ret]

        if len(valid_frames) != len(caps):
            print("오류: 일부 웹캠에서 프레임을 가져올 수 없습니다.")
            break

        for i, frame in enumerate(valid_frames):
            cv2.imshow(f"Webcam ID: {WEBCAM_INDICES[i]}", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("프로그램을 종료합니다.")
            break
        
        elif key == 32: # 스페이스바
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\n[{timestamp}] 이미지 캡처 및 재구성 시작:")

            # 기존 이미지 삭제
            for f in glob.glob(os.path.join(SAVE_DIRECTORY, "*")):
                os.remove(f)
            print("  - 기존 이미지를 모두 삭제했습니다.")

            # 새 이미지 저장
            for i, frame in enumerate(valid_frames):
                image_filename = f"{timestamp}_cam{WEBCAM_INDICES[i]}.png"
                image_save_path = os.path.join(SAVE_DIRECTORY, image_filename)
                cv2.imwrite(image_save_path, frame)
                print(f"  - 웹캠 {WEBCAM_INDICES[i]} 이미지를 '{image_save_path}'에 저장했습니다.")
            
            # 재구성 실행 (기본값 사용)
            perform_reconstruction(FIXED_TARGET_DIR)

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


def update_visualization(target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode):
    # 이 함수는 이제 사용자가 슬라이더 등을 조작할 때만 호출됩니다.
    # 실시간 업데이트는 다른 메커니즘을 사용합니다.
    # 여기서는 간단히 최신 glb 파일을 반환하도록 유지할 수 있습니다.
    if LATEST_GLB_FILE and os.path.exists(LATEST_GLB_FILE):
        return LATEST_GLB_FILE, "Visualization updated by user."
    return None, "No reconstruction available."


# --- [추가 2] 주기적으로 새 결과물을 확인하고 UI를 업데이트하는 함수 ---
# --- [수정 1] 주기적으로 새 결과물을 확인하고 UI를 업데이트하는 함수 (제너레이터로 변경) ---
def live_updater_generator():
    """
    무한 루프를 돌면서 새로운 재구성 결과가 있는지 확인하고,
    결과물을 'yield'하여 뷰어를 업데이트하는 제너레이터 함수입니다.
    """
    while True:
        # 이벤트가 설정될 때까지 최대 0.1초 대기
        if new_reconstruction_available.wait(timeout=0.1):
            new_reconstruction_available.clear() # 이벤트 초기화

            image_paths = sorted(glob.glob(os.path.join(SAVE_DIRECTORY, "*")))
            log_msg = f"Live reconstruction updated at {datetime.now().strftime('%H:%M:%S')}"
            print(f"UI Update Triggered: New file '{LATEST_GLB_FILE}'")

            # 3D 뷰어, 로그, 이미지 갤러리를 한 번에 업데이트하기 위해 값을 yield
            yield LATEST_GLB_FILE, log_msg, image_paths
        else:
            # 업데이트할 내용이 없으면 gr.update()를 yield하여 아무것도 변경하지 않음
            yield gr.update(), gr.update(), gr.update()
        
        # CPU 사용량을 줄이기 위해 약간의 대기 시간 추가
        time.sleep(0.4)

# --- 앱 로드 시 초기 재구성 수행 함수 ---
def reconstruct_on_load():
    """
    Gradio 앱이 로드될 때 고정된 디렉토리에서 초기 재구성을 수행합니다.
    """
    print("Starting initial reconstruction on app load...")
    glbfile, log_msg, image_paths = perform_reconstruction(FIXED_TARGET_DIR)

    if glbfile is None:
        log_msg = f"Error: Initial reconstruction failed. No images found in {SAVE_DIRECTORY}?"
        return None, log_msg, FIXED_TARGET_DIR, gr.Dropdown(choices=["All"]), []

    frame_filter_choices = ["All"] + [f"{i}: {os.path.basename(f)}" for i, f in enumerate(image_paths)]
    return glbfile, log_msg, FIXED_TARGET_DIR, gr.Dropdown(choices=frame_filter_choices, value="All"), image_paths


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(theme=theme, css="""...""") as demo: # CSS는 동일하므로 생략
    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p><strong>실시간 3D 재구성 데모</strong></p>
    <h3>사용법:</h3>
    <ol>
        <li><strong>초기 재구성:</strong> 앱이 로드되면 <code>/home/etrlab/vggt/images</code> 폴더의 이미지로 초기 3D 모델을 생성합니다.</li>
        <li><strong>실시간 재구성:</strong> 별도의 창에 웹캠 화면이 나타납니다. <strong>스페이스바를 누르면</strong> 현재 웹캠 화면을 캡처하여 즉시 3D 재구성을 수행하고 우측 뷰어를 업데이트합니다.</li>
        <li><strong>시각화 조정:</strong> 하단의 옵션을 사용하여 현재 표시된 3D 모델의 시각화 옵션을 변경할 수 있습니다. (이것은 재구성을 다시 수행하지 않습니다)</li>
    </ol>
    </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("**입력 이미지**")
            image_gallery = gr.Gallery(
                label="Preview",
                columns=2, # 2개씩 보이도록 수정
                height="auto",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
            log_output = gr.Markdown("App is loading, starting reconstruction...", elem_classes=["custom-log"])
            reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5, label="3D Viewer")
            
            with gr.Accordion("Visualization Controls", open=False):
                with gr.Row():
                    prediction_mode = gr.Radio(["Depthmap and Camera Branch", "Pointmap Branch"], label="Prediction Mode", value="Depthmap and Camera Branch")

                with gr.Row():
                    conf_thres = gr.Slider(minimum=0, maximum=100, value=10, step=0.1, label="Confidence Threshold (%)")
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                    with gr.Column():
                        show_cam = gr.Checkbox(label="Show Camera", value=True)
                        mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                        mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                        mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # 시각화 옵션 변경 시 UI 업데이트 (재계산 없음)
    controls_to_update = [conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode]
    for control in controls_to_update:
        # 이 부분은 현재 버전에서는 완전한 재시각화를 수행하지는 않지만,
        # 향후 npz 파일을 로드하여 재시각화하는 로직으로 확장할 수 있습니다.
        # 지금은 단순화를 위해 비워둡니다.
        pass
# --- 앱 로드 시 초기 재구성 실행 ---
    demo.load(
        fn=reconstruct_on_load,
        inputs=[],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
    )

    # --- [수정 2] 주기적 업데이트 로직 변경 ---
    # 에러가 발생한 'every' 인자를 사용하는 demo.load를 삭제하고,
    # 새로 만든 제너레이터 함수를 load 이벤트에 연결합니다.
    demo.load(
        fn=live_updater_generator,  # 제너레이터 함수로 교체
        inputs=None,
        outputs=[reconstruction_output, log_output, image_gallery]
        # 'every' 인자 삭제
    )

    # --- 백그라운드에서 웹캠 스레드 시작 ---
    webcam_thread = threading.Thread(target=opencv_capture_and_reconstruct, daemon=True)
    webcam_thread.start()

    demo.queue(max_size=20).launch(show_error=True, share=True)
