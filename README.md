# Freeway Speed

Freeway Speed 是一個面向低畫質公路 CCTV 的研究型專案，目標是在彎道路段下，完成車輛偵測、跨幀追蹤、距離估算與速度估算。

這個專案特別針對「畫質差、壓縮重、角度不固定」的監視器情境，嘗試以開源預訓練模型搭配幾何方法，建立一條可落地、可觀測、可持續改進的流程。

> [!IMPORTANT]
> 此專案為實驗性原型，精度、實作完整度與泛化能力皆有待提升。
>
> - 不同攝影機角度、畫質、天候與壓縮率會明顯影響結果。
> - 目前比例尺估算採用啟發式（虛線 + 車道寬度），非嚴格標定流程。
> - 請勿將輸出直接用於執法、裁罰或高風險決策情境。

## 專案亮點

- 面向低畫質 CCTV 的偵測與追蹤參數調整
- 以動態 IPM 將視角轉為 BEV，再進行曲線幾何計算
- 弧長積分估算彎道距離，而非單純平面直線距離
- 同步輸出影片與 CSV，保留每幀分析資料，便於診斷
- 比例尺來源可追蹤（`dashed` / `lane_width` / `default`）

## 系統流程（預設）

- 車輛偵測：YOLOv8 (`yolov8n.pt`)
- 車道線分割：YOLOPv2 ONNX (`yolop-640-640.onnx`)
- 追蹤：ByteTrack
- 幾何：動態 IPM + BEV 曲線弧長積分
- 比例尺：虛線法 (`dashed`) 優先，失敗時改車道寬度法 (`lane_width`)，最後才 fallback (`default`)

## 快速開始

### 1) 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) 執行

方式 A：安裝後使用 CLI 指令

```bash
freeway-speed \
 --config configs/default.yaml \
 --input /path/to/video.mp4 \
 --output outputs/result.mp4
```

方式 B：使用 Python module

```bash
python -m freeway_speed \
 --config configs/default.yaml \
 --input /path/to/video.mp4 \
 --output outputs/result.mp4
```

上面指令會同步輸出 `outputs/result.csv`。

即時顯示：使用 `--display` 參數可以在處理過程中彈出視窗顯示疊圖結果，便於調參與診斷。

```bash
freeway-speed \
 --config configs/default.yaml \
 --input /path/to/video.mp4 \
 --display \
 --output outputs/result.mp4
```

自訂紀錄檔路徑：使用 `--log-output` 參數可以指定 CSV 輸出路徑與檔名。

```bash
freeway-speed \
 --config configs/default.yaml \
 --input /path/to/video.mp4 \
 --output outputs/result.mp4 \
 --log-output outputs/result-log.csv
```

## 輸出資料說明

### 影片輸出

- `--output` 指定處理後影片路徑

### CSV 輸出

- 預設會產生與輸出影片同名 `.csv` 檔
- 可用 `--log-output` 指定路徑

主要欄位：

- `frame_idx`, `video_time_sec`, `proc_timestamp_sec`
- `track_id`, `class_name`, `score`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `bev_x`, `bev_y`
- `distance_m`, `speed_kmh`
- `scale_m_per_px`, `scale_source` (`dashed` / `lane_width` / `default`)
- `lane_a`, `lane_b`, `lane_c`

## 主要模組

- `src/freeway_speed/perception.py`: YOLO 車輛偵測與 ONNX 車道分割
- `src/freeway_speed/ipm.py`: 動態逆透視轉換
- `src/freeway_speed/curve.py`: BEV 車道曲線擬合
- `src/freeway_speed/geometry.py`: 弧長積分與比例尺估算
- `src/freeway_speed/tracking.py`: ByteTrack 封裝
- `src/freeway_speed/speed.py`: 速度估算與平滑
- `src/freeway_speed/pipeline.py`: 端到端整合與疊圖
- `src/freeway_speed/cli.py`: CLI 與 CSV 輸出

## 調參方向

- 偵測漏抓：降低 `perception.yolo_conf`，提高 `perception.yolo_imgsz`
- 追蹤不連續：調大 `tracking.bt_track_buffer`，微調 `tracking.bt_match_thresh`
- 速度抖動：調大 `tracking.min_dt_sec` 與 `tracking.speed_smooth_window`
- 比例尺不穩：調整 `calibration.search_band_px`、`calibration.lane_hist_roi_start_ratio`
- 比例尺常 fallback：優先檢查車道線可視度，再調 `calibration.lane_peak_min_votes`

## 專案定位

此專案目前定位為「研究與工程驗證工具」，適合：

- 驗證低畫質 CCTV 的可行性
- 快速比較不同模型/參數組合
- 建立後續標定與精度提升的基線
