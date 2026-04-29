"""Render a paper-style Fig. 1 teaser from a real exported demo case."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence


ROOT = Path("/home/rl182/dl/V2L/Project-meme/MultiSticker")
DEMO_DIR = ROOT / "Latex_report" / "demo_assets" / "dual_lora_png_demo"
OUT_DIR = ROOT / "Latex_report" / "img"
FONT = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
BOLD = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD if bold else FONT), size=size)


def ffmpeg_bin() -> str | None:
    for item in [
        os.environ.get("FFMPEG_BIN"),
        shutil.which("ffmpeg"),
        "/opt/apps/software/FFmpeg/7.0.2-GCCcore-13.3.0/bin/ffmpeg",
        "/opt/apps/software/FFmpeg/7.0.2-GCCcore-13.2.0/bin/ffmpeg",
    ]:
        if item and Path(item).exists():
            return item
    return None


def preview(src: Path, size: int = 180) -> Image.Image:
    suffix = src.suffix.lower()
    try:
        if suffix == ".png":
            frame = Image.open(src).convert("RGBA")
        elif suffix == ".gif":
            with Image.open(src) as im:
                frame = next(ImageSequence.Iterator(im)).convert("RGBA")
        elif suffix == ".webm" and ffmpeg_bin():
            tmp = src.with_suffix(".fig1_first_frame.png")
            subprocess.run(
                [ffmpeg_bin(), "-y", "-i", str(src), "-frames:v", "1", str(tmp)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            frame = Image.open(tmp).convert("RGBA")
        else:
            frame = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    except Exception:
        frame = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    frame.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    canvas.alpha_composite(frame, ((size - frame.width) // 2, (size - frame.height) // 2))
    return canvas


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill, outline=None, radius: int = 22, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def parse_turns(text: str, limit: int = 5) -> list[tuple[str, str]]:
    turns = []
    for raw in str(text).split("||"):
        raw = " ".join(raw.split())
        if not raw:
            continue
        speaker, msg = raw.split(":", 1) if ":" in raw else ("speaker", raw)
        msg = msg.strip().replace("[sticker:", "[sticker ")
        if len(msg) > 105:
            msg = msg[:102] + "..."
        turns.append((speaker.strip() or "speaker", msg or "[empty message]"))
    return turns[-limit:]


def paste_sticker(base: Image.Image, img: Image.Image, xy: tuple[int, int], box: int) -> None:
    x, y = xy
    shadow = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((8, 10, box - 2, box - 2), radius=24, fill=(0, 0, 0, 28))
    base.alpha_composite(shadow, (x, y))
    card = Image.new("RGBA", (box, box), (255, 255, 255, 255))
    cd = ImageDraw.Draw(card)
    cd.rounded_rectangle((0, 0, box - 10, box - 12), radius=24, fill=(255, 255, 255, 255), outline=(216, 222, 232), width=2)
    card.alpha_composite(img, ((box - 10 - img.width) // 2, (box - 12 - img.height) // 2))
    base.alpha_composite(card, (x, y))


def media_file(case_dir: Path, prefix: str, sticker_id: str) -> Path:
    return next(case_dir.glob(f"{prefix}_{Path(sticker_id).stem}.*"))


def render(case_dir: Path, out_path: Path) -> None:
    with open(case_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    w, h = 2400, 1350
    img = Image.new("RGBA", (w, h), (248, 250, 252, 255))
    draw = ImageDraw.Draw(img)
    title_f = font(54, True)
    h1 = font(34, True)
    body = font(27)
    small = font(21)
    tiny = font(18)

    draw.text((90, 52), "MultiSticker retrieves the right sticker from real dialogue memory", font=title_f, fill=(14, 24, 39))
    draw.text((92, 120), "Real chat context + retrieved memory + reply intent -> dual-LoRA CLIP ranking over animated sticker media", font=body, fill=(69, 79, 92))

    phone = (90, 205, 780, 1195)
    rounded(draw, phone, fill=(255, 255, 255, 255), outline=(207, 216, 228), radius=40, width=3)
    draw.text((130, 245), "Dialogue context", font=h1, fill=(14, 24, 39))
    y = 310
    for idx, (speaker, msg) in enumerate(parse_turns(meta.get("context_preview", ""))):
        left_side = idx % 2 == 0
        bubble_w = 500
        x = 130 if left_side else 235
        lines = wrap(draw, msg, small, bubble_w - 44)[:3]
        bubble_h = 58 + len(lines) * 28
        fill = (240, 244, 248, 255) if left_side else (224, 242, 254, 255)
        rounded(draw, (x, y, x + bubble_w, y + bubble_h), fill=fill, outline=None, radius=26)
        draw.text((x + 24, y + 14), speaker, font=tiny, fill=(71, 85, 105))
        for line_i, line in enumerate(lines):
            draw.text((x + 24, y + 42 + line_i * 29), line, font=small, fill=(15, 23, 42))
        y += bubble_h + 22

    mid_x = 875
    draw.text((mid_x, 245), "Structure used by the retriever", font=h1, fill=(14, 24, 39))
    streams = [
        ("Local context", "Last turns from the current chat session"),
        ("Long-term memory", meta.get("memory_preview", "")[:150] + "..."),
        ("Reply intent", f"{meta.get('intent_label', '')}: {meta.get('intent_text', '')}"),
    ]
    y = 320
    colors = [(236, 253, 245, 255), (239, 246, 255, 255), (254, 249, 195, 255)]
    for i, (name, desc) in enumerate(streams):
        rounded(draw, (mid_x, y, mid_x + 620, y + 150), fill=colors[i], outline=(203, 213, 225), radius=28)
        draw.text((mid_x + 28, y + 22), name, font=font(29, True), fill=(15, 23, 42))
        for line_i, line in enumerate(wrap(draw, desc, small, 550)[:3]):
            draw.text((mid_x + 28, y + 64 + line_i * 30), line, font=small, fill=(51, 65, 85))
        y += 185
    rounded(draw, (mid_x + 85, y + 25, mid_x + 535, y + 160), fill=(15, 23, 42, 255), radius=34)
    draw.text((mid_x + 128, y + 60), "Dual-LoRA OpenCLIP", font=font(31, True), fill=(255, 255, 255))
    draw.text((mid_x + 138, y + 98), "query projected into sticker space", font=small, fill=(203, 213, 225))

    for line_y in [395, 580, 765]:
        draw.line((mid_x + 620, line_y, mid_x + 705, 692), fill=(100, 116, 139), width=5)
    draw.line((mid_x + 620, y + 95, 1600, y + 95), fill=(100, 116, 139), width=5)

    right_x = 1605
    draw.text((right_x, 245), "Ranked sticker outputs", font=h1, fill=(14, 24, 39))
    gold_path = media_file(case_dir, "gold", meta["gold"])
    paste_sticker(img, preview(gold_path, 210), (right_x, 320), 250)
    draw.text((right_x + 275, 344), "Observed reply", font=font(31, True), fill=(14, 24, 39))
    draw.text((right_x + 275, 388), f"gold rank {meta.get('gold_rank')}  score {meta.get('gold_score')}", font=body, fill=(51, 65, 85))
    draw.text((right_x + 275, 430), Path(meta["gold"]).stem[:24], font=small, fill=(100, 116, 139))

    y = 615
    for pred in meta["top_predictions"][:5]:
        path = media_file(case_dir, f"top{pred['rank']}", pred["sticker_id"])
        x = right_x + (pred["rank"] - 1) * 148
        paste_sticker(img, preview(path, 112), (x, y), 132)
        draw.text((x + 16, y + 142), f"Top {pred['rank']}", font=font(22, True), fill=(15, 23, 42))
        draw.text((x + 8, y + 171), f"{pred['score']:.2f}", font=tiny, fill=(71, 85, 105))

    draw.text((right_x, 1015), "The model sees conversation, memory, and intent together,", font=small, fill=(51, 65, 85))
    draw.text((right_x, 1055), "then ranks the same sticker bank a user would browse.", font=small, fill=(51, 65, 85))
    draw.text((right_x, 1132), f"Case sample: {meta.get('sample_id', '')}", font=tiny, fill=(100, 116, 139))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(out_path, quality=95)
    print(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case_02_rank_2")
    parser.add_argument("--out", default=str(OUT_DIR / "fig1_multisticker_teaser.png"))
    args = parser.parse_args()
    render(DEMO_DIR / args.case, Path(args.out))


if __name__ == "__main__":
    main()
