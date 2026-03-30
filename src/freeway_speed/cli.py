from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from .config import load_config
from .pipeline import FreewaySpeedPipeline, draw_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-quality CCTV freeway speed estimator")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="", help="Output video path")
    parser.add_argument(
        "--log-output",
        type=str,
        default="",
        help="CSV log path (default: sidecar next to --output or input name)",
    )
    parser.add_argument("--display", action="store_true", help="Display processed frames")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 15.0
    pipeline = FreewaySpeedPipeline(cfg, frame_rate=int(round(fps)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if args.log_output:
        log_path = Path(args.log_output)
    elif args.output:
        out_path = Path(args.output)
        log_path = out_path.with_suffix(".csv")
    else:
        log_path = Path(args.input).with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8", newline="")
    log_writer = csv.writer(log_fp)
    log_writer.writerow(
        [
            "frame_idx",
            "video_time_sec",
            "proc_timestamp_sec",
            "track_id",
            "class_name",
            "score",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bev_x",
            "bev_y",
            "distance_m",
            "speed_kmh",
            "scale_m_per_px",
            "scale_source",
            "lane_a",
            "lane_b",
            "lane_c",
        ]
    )

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms > 0:
            video_t = pos_ms / 1000.0
        else:
            video_t = frame_count / fps
        proc_ts = time.perf_counter()
        state = pipeline.process_frame(frame, video_t)
        vis = draw_overlay(frame, state)

        lane_a = state.lane_poly.a if state.lane_poly is not None else None
        lane_b = state.lane_poly.b if state.lane_poly is not None else None
        lane_c = state.lane_poly.c if state.lane_poly is not None else None
        for obj in state.tracked:
            x1, y1, x2, y2 = obj.bbox
            bev_x = obj.bev_point[0] if obj.bev_point is not None else None
            bev_y = obj.bev_point[1] if obj.bev_point is not None else None
            log_writer.writerow(
                [
                    frame_count,
                    video_t,
                    proc_ts,
                    obj.track_id,
                    obj.class_name,
                    obj.score,
                    x1,
                    y1,
                    x2,
                    y2,
                    bev_x,
                    bev_y,
                    obj.distance_m,
                    obj.speed_kmh,
                    state.scale_m_per_px,
                    state.scale_source,
                    lane_a,
                    lane_b,
                    lane_c,
                ]
            )

        if not state.tracked:
            log_writer.writerow(
                [
                    frame_count,
                    video_t,
                    proc_ts,
                    -1,
                    "",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    state.scale_m_per_px,
                    state.scale_source,
                    lane_a,
                    lane_b,
                    lane_c,
                ]
            )

        if writer is not None:
            writer.write(vis)

        if args.display:
            cv2.imshow("freeway-speed", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    log_fp.close()
    if args.display:
        cv2.destroyAllWindows()

    print(f"CSV log saved to: {log_path}")


if __name__ == "__main__":
    run()
