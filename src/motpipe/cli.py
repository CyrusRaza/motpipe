from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .detectors import DETECTORS
from .pipeline import create_pipeline
from .sources import VideoWriter, open_source
from .trackers import TRACKERS
from .visualize import draw_fps, draw_tracks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motpipe",
        description="Multi-object detection + tracking with interchangeable strategies.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", help="Video file path, or a webcam index such as 0")
    src.add_argument("--image-dir", help="Folder of frames, played in filename order")
    p.add_argument("--detector", default="yolov8", choices=DETECTORS.names())
    p.add_argument("--tracker", default="bytetrack", choices=TRACKERS.names())
    p.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, ...")
    p.add_argument("--fps", type=float, default=30.0, help="Frame rate for --image-dir input")
    p.add_argument("--output", type=Path, help="Output video path (.mp4)")
    p.add_argument("--show", action="store_true", help="Display a live window (press q to quit)")
    p.add_argument("--list", action="store_true", help="List available detectors and trackers")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("detectors:", ", ".join(DETECTORS.names()))
        print("trackers: ", ", ".join(TRACKERS.names()))
        return 0
    if not (args.video or args.image_dir):
        parser.error("one of --video or --image-dir is required")
    if not (args.output or args.show):
        parser.error("nothing to do: pass --output and/or --show")

    import cv2

    try:
        source = open_source(args.video, args.image_dir, args.fps)
    except OSError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    pipeline = create_pipeline(
        args.detector, args.tracker, conf=args.conf, device=args.device, frame_rate=source.fps,
        on_status=lambda msg: print(f"[INFO] {msg}"),
    )
    writer = VideoWriter(args.output, source.fps, source.size) if args.output else None

    count = 0
    try:
        for result in pipeline.run(source):
            draw_tracks(result.frame, result.tracks)
            draw_fps(result.frame, result.fps)
            if writer:
                writer.write(result.frame)
            if args.show:
                cv2.imshow("motpipe", result.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            count += 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        source.close()
        if writer:
            writer.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"[INFO] Processed {count} frames" + (f" -> {args.output}" if args.output else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
