"""Build CVPR-style single-column demo PDFs for exported dual-LoRA cases.

The renderer expects each case directory to contain:
- ``metadata.json`` with full ``context_text`` (sticker tokens preserved)
- ``gold_<id>``, ``top1_<id>`` ... ``top5_<id>`` media files
- optional ``context_stickers/<id>`` files for inline rendering

Inline sticker tokens (``[sticker:xxx.png]``) inside ``context_text`` are
replaced by small thumbnail images. Predictions are laid out as one large
gold sticker followed by a row of five candidate thumbnails, with a green
or red border indicating same/different intent group.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageSequence


ROOT = Path("/home/rl182/dl/V2L/Project-meme/MultiSticker")
DEMO_DIR = ROOT / "Latex_report" / "demo_assets" / "dual_lora_png_demo"
PDFLATEX = Path("/scratch/rl182/tex/.TinyTeX/bin/x86_64-linux/pdflatex")
PDFCROP = Path("/scratch/rl182/tex/.TinyTeX/bin/x86_64-linux/pdfcrop")

STICKER_TOKEN_RE = re.compile(r"\[sticker:([^\]]+)\]")


def latex_escape(value: object) -> str:
    text = str(value).encode("ascii", "ignore").decode("ascii")
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def find_ffmpeg() -> str | None:
    candidates = [
        os.environ.get("FFMPEG_BIN"),
        shutil.which("ffmpeg"),
        "/opt/apps/software/FFmpeg/7.0.2-GCCcore-13.3.0/bin/ffmpeg",
        "/opt/apps/software/FFmpeg/7.0.2-GCCcore-13.2.0/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def make_preview(src: Path, dst: Path, ffmpeg_bin: str | None) -> Path:
    """Convert any sticker media file to a single-frame PNG preview."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    suffix = src.suffix.lower()
    if suffix == ".png":
        try:
            with Image.open(src) as im:
                im.convert("RGBA").save(dst)
            return dst
        except Exception:
            shutil.copy2(src, dst)
            return dst
    if suffix == ".gif":
        with Image.open(src) as im:
            frame = next(ImageSequence.Iterator(im)).convert("RGBA")
            frame.save(dst)
        return dst
    if suffix == ".webm" and ffmpeg_bin:
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(src), "-frames:v", "1", str(dst)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return dst
    canvas = Image.new("RGBA", (256, 256), (255, 255, 255, 0))
    canvas.save(dst)
    return dst


def case_sort_key(path: Path) -> int:
    return int(path.name.split("_")[1])


def list_case_dirs() -> list[Path]:
    return sorted(DEMO_DIR.glob("case_*_rank_*"), key=case_sort_key)


def select_case_dirs(case_range: str) -> list[Path]:
    dirs = list_case_dirs()
    if case_range == "all":
        return dirs
    wanted: set[int] = set()
    for part in case_range.split(","):
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            wanted.update(range(start, end + 1))
        else:
            wanted.add(int(part))
    return [path for path in dirs if case_sort_key(path) in wanted]


def parse_turn(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    if not raw:
        return "", ""
    # split only on the first colon followed by space, but speaker IDs in this
    # corpus don't contain spaces so the simple "first colon" rule is fine.
    if ":" in raw:
        speaker, message = raw.split(":", 1)
        return speaker.strip(), message.strip()
    return "speaker", raw


def split_turns(context_text: str, limit: int = 8) -> list[tuple[str, str]]:
    turns = []
    for raw in str(context_text).split("||"):
        speaker, message = parse_turn(raw)
        if not message and not speaker:
            continue
        turns.append((speaker, message))
    return turns[-limit:]


def render_inline_message(
    message: str,
    context_sticker_dir: Path,
    preview_dir: Path,
    base_dir: Path,
    ffmpeg_bin: str | None,
    sticker_only_size: str = "0.85in",
    inline_height: str = "4em",
) -> tuple[str, bool]:
    """Replace [sticker:xxx] tokens with \\includegraphics, escape the rest.

    Returns (rendered, sticker_only) where sticker_only indicates the message
    contains only sticker tokens (no surrounding text) — caller may choose to
    render without a chat bubble.
    """
    parts: list[str] = []
    last = 0
    text_chars = 0
    sticker_count = 0
    for m in STICKER_TOKEN_RE.finditer(message):
        if m.start() > last:
            chunk = message[last:m.start()]
            text_chars += len(chunk.strip())
            parts.append(latex_escape(chunk))
        sid = m.group(1).strip()
        sticker_count += 1
        sticker_file = context_sticker_dir / sid
        if sticker_file.exists():
            preview = make_preview(
                sticker_file,
                preview_dir / f"ctx_{Path(sid).stem}.png",
                ffmpeg_bin,
            )
            rel = os.path.relpath(preview, base_dir).replace(os.sep, "/")
            parts.append(
                "{\\raisebox{-0.32\\height}{"
                f"\\includegraphics[height={inline_height}]{{{rel}}}"
                "}}"
            )
        else:
            parts.append(r"{\scriptsize\textsf{[sticker]}}")
        last = m.end()
    if last < len(message):
        chunk = message[last:]
        text_chars += len(chunk.strip())
        parts.append(latex_escape(chunk))
    sticker_only = sticker_count > 0 and text_chars == 0
    if sticker_only:
        # Re-render at standalone size, using the same files but bigger.
        sticker_parts: list[str] = []
        for m in STICKER_TOKEN_RE.finditer(message):
            sid = m.group(1).strip()
            sticker_file = context_sticker_dir / sid
            if sticker_file.exists():
                preview = make_preview(
                    sticker_file,
                    preview_dir / f"ctx_{Path(sid).stem}.png",
                    ffmpeg_bin,
                )
                rel = os.path.relpath(preview, base_dir).replace(os.sep, "/")
                sticker_parts.append(
                    f"\\includegraphics[height={sticker_only_size}]{{{rel}}}"
                )
            else:
                sticker_parts.append(r"{\scriptsize\textsf{[sticker]}}")
        return "\\,".join(sticker_parts), True
    text = "".join(parts).strip()
    return (text or r"{\scriptsize\itshape (empty)}"), False


AVATAR_PALETTE = [
    "Apricot", "SkyBlue", "Lavender", "YellowGreen", "Salmon",
    "Goldenrod", "Aquamarine", "Orchid", "Tan", "CornflowerBlue",
]


def avatar_color(speaker: str) -> str:
    if not speaker:
        return AVATAR_PALETTE[0]
    return AVATAR_PALETTE[sum(ord(c) for c in speaker) % len(AVATAR_PALETTE)]


def avatar_initials(speaker: str) -> str:
    s = (speaker or "?").strip()
    return s[:2].upper() if s else "??"


def render_dialogue(
    context_text: str,
    context_sticker_dir: Path,
    preview_dir: Path,
    base_dir: Path,
    ffmpeg_bin: str | None,
    limit: int = 8,
) -> str:
    """Render the chat as a phone-style messaging UI."""
    turns = split_turns(context_text, limit=limit)
    if not turns:
        return r"{\itshape (no dialogue)}"

    # The last speaker is treated as "You" (right-aligned, blue iMessage bubble).
    me_speaker = turns[-1][0]

    rendered: list[str] = []
    for speaker, message in turns:
        rendered_msg, sticker_only = render_inline_message(
            message, context_sticker_dir, preview_dir, base_dir, ffmpeg_bin
        )
        is_me = speaker == me_speaker
        avatar_label = "@you" if is_me else f"@{speaker[:8]}" if speaker else "@user"
        ainit = avatar_initials(speaker)
        acolor = "IGBlue" if is_me else avatar_color(speaker)
        avatar = (
            f"\\tikz[baseline=-0.6ex]{{\\node[circle,fill={acolor},inner sep=0pt,"
            f"minimum size=4.5mm,font=\\scriptsize\\bfseries\\color{{white}}] {{{ainit}}};}}"
        )
        if sticker_only:
            content = (
                f"{{\\tiny\\color{{gray!70!black}}\\textsf{{{latex_escape(avatar_label)}}}}}\\par"
                f"{rendered_msg}"
            )
        elif is_me:
            content_inner = (
                "\\begin{tcolorbox}[imebubble]"
                f"{rendered_msg}"
                "\\end{tcolorbox}"
            )
            content = (
                f"{{\\tiny\\color{{gray!70!black}}\\textsf{{{latex_escape(avatar_label)}}}}}\\par"
                f"{content_inner}"
            )
        else:
            content_inner = (
                "\\begin{tcolorbox}[otherbubble]"
                f"{rendered_msg}"
                "\\end{tcolorbox}"
            )
            content = (
                f"{{\\tiny\\color{{gray!60!black}}\\textsf{{{latex_escape(avatar_label)}}}}}\\par"
                f"{content_inner}"
            )

        if is_me:
            row = (
                "\\par\\vspace{0.8mm}\\noindent\\hspace*{\\fill}"
                "\\begin{minipage}{0.74\\linewidth}\n"
                "\\raggedleft\n"
                f"{content}\n"
                "\\end{minipage}"
                f"\\,{avatar}\\par"
            )
        else:
            row = (
                "\\par\\vspace{0.8mm}\\noindent"
                f"{avatar}\\,"
                "\\begin{minipage}{0.74\\linewidth}\n"
                f"{content}\n"
                "\\end{minipage}\\par"
            )
        rendered.append(row)

    body = "\n".join(rendered)
    return (
        "\\begin{tcolorbox}[phoneframe]\n"
        # status bar mock
        "{\\scriptsize\\sffamily\\color{gray!50!black}\\textbf{9:41}\\hfill"
        "\\textbullet\\,\\textbullet\\,\\textbullet\\,\\textbullet\\hfill"
        "\\textbf{100\\%}}\\\\[0.6mm]\n"
        "{\\centering\\small\\sffamily\\bfseries Group chat\\par}\n"
        "\\vspace{1mm}\n"
        f"{body}\n"
        "\\end{tcolorbox}"
    )


def media_for_case(
    case_dir: Path, metadata: dict
) -> list[tuple[str, str, float | None, bool | None, Path]]:
    items: list[tuple[str, str, float | None, bool | None, Path]] = []
    gold = metadata["gold"]
    gold_file = next(case_dir.glob(f"gold_{Path(gold).stem}.*"))
    items.append(("Gold", gold, metadata.get("gold_score"), True, gold_file))
    for pred in metadata["top_predictions"][:5]:
        sticker_id = pred["sticker_id"]
        media_file = next(case_dir.glob(f"top{pred['rank']}_{Path(sticker_id).stem}.*"))
        items.append(
            (
                f"Top {pred['rank']}",
                sticker_id,
                pred.get("score"),
                pred.get("same_group"),
                media_file,
            )
        )
    return items


def render_predictions_block(
    case_dir: Path, metadata: dict, base_dir: Path, ffmpeg_bin: str | None,
    column_mode: bool = False,
) -> str:
    items = media_for_case(case_dir, metadata)
    gold_label, gold_id, gold_score, _, gold_file = items[0]
    gold_preview = make_preview(
        gold_file,
        case_dir / "previews" / f"{gold_file.stem}_first_frame.png",
        ffmpeg_bin,
    )
    gold_rel = os.path.relpath(gold_preview, base_dir).replace(os.sep, "/")
    gold_score_str = "" if gold_score is None else f"{gold_score:.3f}"

    cell_w = "0.155" if column_mode else "0.185"
    cells = []
    for label, sticker_id, score, same_group, media_file in items[1:]:
        preview = make_preview(
            media_file,
            case_dir / "previews" / f"{media_file.stem}_first_frame.png",
            ffmpeg_bin,
        )
        rel = os.path.relpath(preview, base_dir).replace(os.sep, "/")
        border = "ForestGreen" if same_group else "BrickRed"
        score_text = "" if score is None else f"{score:.3f}"
        cells.append(
            f"\\begin{{minipage}}[t]{{{cell_w}\\linewidth}}\\centering\n"
            f"\\fcolorbox{{{border}}}{{white}}{{"
            f"\\includegraphics[width=0.92\\linewidth,height=0.92\\linewidth,keepaspectratio]{{{rel}}}"
            f"}}\\\\[0.4mm]\n"
            f"{{\\tiny\\bfseries {latex_escape(label)}}}\\\\[-0.2mm]\n"
            f"{{\\tiny {latex_escape(score_text)}}}\n"
            f"\\end{{minipage}}"
        )
    candidates_row = "\\hfill\n".join(cells)

    if column_mode:
        # Compact horizontal: gold | top1..top5 in single row
        gold_cell = (
            f"\\begin{{minipage}}[t]{{{cell_w}\\linewidth}}\\centering\n"
            "\\fcolorbox{ForestGreen}{white}{"
            f"\\includegraphics[width=0.92\\linewidth,height=0.92\\linewidth,keepaspectratio]{{{gold_rel}}}"
            "}\\\\[0.4mm]\n"
            "{\\tiny\\bfseries\\color{ForestGreen!70!black} Gold}\\\\[-0.2mm]\n"
            f"{{\\tiny {latex_escape(gold_score_str)}}}\n"
            "\\end{minipage}"
        )
        all_row = gold_cell + "\\hfill\n" + candidates_row
        return (
            "\\nopagebreak\n"
            "\\begin{tcolorbox}[stickerpanel]\n"
            "{\\footnotesize\\sffamily\\color{gray!60!black}"
            "\\textbf{Suggested stickers}\\hfill"
            "{\\tiny powered by Dual-LoRA}}\\\\[0.5mm]\n"
            "{\\tiny\\sffamily\\color{gray!60!black}"
            "Gold (\\textcolor{ForestGreen!70!black}{green}) vs.\\ model top-5 "
            "(\\textcolor{BrickRed}{red}: diff intent):}\\\\[0.6mm]\n"
            f"\\noindent {all_row}\n"
            "\\end{tcolorbox}\n"
        )
    return (
        "\\nopagebreak\n"
        "\\begin{tcolorbox}[stickerpanel]\n"
        "{\\footnotesize\\sffamily\\color{gray!60!black}"
        "\\textbf{Suggested stickers}\\hfill"
        "{\\tiny powered by Dual-LoRA retriever}}\\\\[1mm]\n"
        "\\begin{center}\n"
        "\\begin{minipage}{0.20\\linewidth}\\centering\n"
        "\\fcolorbox{ForestGreen}{white}{"
        f"\\includegraphics[width=0.95\\linewidth,height=0.95\\linewidth,keepaspectratio]{{{gold_rel}}}"
        "}\\\\[0.5mm]\n"
        f"{{\\scriptsize\\sffamily\\bfseries\\color{{ForestGreen!70!black}} Gold (ground truth)}}\\\\[-0.2mm]\n"
        f"{{\\tiny\\sffamily\\color{{gray!60!black}} {latex_escape(gold_score_str)}}}\n"
        "\\end{minipage}\n"
        "\\end{center}\n"
        "\\vspace{0.6mm}\n"
        "{\\scriptsize\\sffamily\\color{gray!60!black}"
        "Model top-5 (\\textcolor{ForestGreen!70!black}{\\textbf{green}}: same intent, "
        "\\textcolor{BrickRed}{\\textbf{red}}: different):}\\\\[1mm]\n"
        f"\\noindent {candidates_row}\n"
        "\\end{tcolorbox}\n"
    )


def render_case_block(case_dir: Path, metadata: dict, base_dir: Path, column_mode: bool = False) -> str:
    case_no = case_sort_key(case_dir)
    ffmpeg_bin = find_ffmpeg()
    context_sticker_dir = case_dir / "context_stickers"
    preview_dir = case_dir / "previews"

    context_text = metadata.get("context_text") or metadata.get("context_preview", "")
    dialogue = render_dialogue(
        context_text,
        context_sticker_dir,
        preview_dir,
        base_dir,
        ffmpeg_bin,
        limit=4 if column_mode else 6,
    )
    raw_memory = str(metadata.get("memory_text") or metadata.get("memory_preview", ""))
    memory_segments = [seg.strip() for seg in raw_memory.split("||") if seg.strip()]
    memory_text = memory_segments[-1] if memory_segments else ""
    if len(memory_text) > 240:
        memory_text = memory_text[:237] + "..."
    memory = latex_escape(" ".join(memory_text.split()))
    intent_label = latex_escape(metadata.get("intent_label", ""))
    intent_text = latex_escape(metadata.get("intent_text", ""))
    sample_id = latex_escape(metadata.get("sample_id", ""))
    domain = latex_escape(metadata.get("domain", ""))
    gold_rank = metadata.get("gold_rank", "?")

    predictions = render_predictions_block(case_dir, metadata, base_dir, ffmpeg_bin, column_mode=column_mode)

    inner_width = "\\linewidth" if column_mode else "0.62\\linewidth"
    return rf"""
\noindent{{\scriptsize\sffamily\textbf{{Case {case_no:02d}}}\,\color{{gray!60!black}}$\cdot$\,{domain}\,$\cdot$\,\texttt{{{intent_label}}}\,$\cdot$\,Gold rank: {gold_rank}}}

\vspace{{0.8mm}}
\begin{{center}}
\begin{{minipage}}{{{inner_width}}}
{dialogue}

\vspace{{1mm}}
{predictions}
\end{{minipage}}
\end{{center}}

\vspace{{0.6mm}}
\noindent{{\scriptsize\sffamily\textbf{{\color{{gray!50!black}}Session memory:}} \itshape\color{{gray!40!black}} {memory}}}\\[0.2mm]
\noindent{{\scriptsize\sffamily\textbf{{\color{{gray!50!black}}Reply intent:}} \color{{gray!40!black}} {intent_text}}}
"""


def document(body: str, title: str, column: bool = False) -> str:
    if column:
        docclass = "\\documentclass[varwidth=3.25in,border={3pt 3pt 3pt 3pt},9pt]{standalone}"
        geometry = ""
    else:
        docclass = "\\documentclass[10pt,letterpaper]{article}"
        geometry = "\\usepackage[margin=0.75in]{geometry}"
    title_block = "" if column else (
        "\\begin{center}\n"
        f"{{\\large\\bfseries\\sffamily {latex_escape(title)}}}\\\\[2pt]\n"
        "{\\footnotesize\\itshape Dual-LoRA dialogue-conditioned sticker retrieval}\n"
        "\\end{center}\n\\vspace{1mm}\n"
    )
    return rf"""{docclass}
{geometry}
\usepackage{{graphicx}}
\usepackage[dvipsnames]{{xcolor}}
\usepackage{{mathptmx}}
\usepackage[T1]{{fontenc}}
\usepackage{{tikz}}
\usepackage[most]{{tcolorbox}}
\usepackage{{hyperref}}
\hypersetup{{hidelinks}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{2pt}}
\renewcommand{{\baselinestretch}}{{1.05}}

% --- chat bubble + phone styles ---
\definecolor{{IGBlue}}{{RGB}}{{0,132,255}}
\definecolor{{IGBubbleGray}}{{RGB}}{{235,235,238}}
\definecolor{{PhoneBG}}{{RGB}}{{248,249,251}}
\definecolor{{PhoneBorder}}{{RGB}}{{210,212,218}}

\tcbset{{
  phoneframe/.style={{
    enhanced, colback=PhoneBG, colframe=PhoneBorder,
    arc=4mm, boxrule=0.6pt, left=2.5mm, right=2.5mm, top=1.5mm, bottom=1.5mm,
    width=\linewidth, drop fuzzy shadow=gray!30,
  }},
  otherbubble/.style={{
    colback=IGBubbleGray, colframe=IGBubbleGray,
    arc=2.6mm, boxrule=0pt, left=2.2mm, right=2.2mm, top=1mm, bottom=1mm,
    width=\linewidth, before skip=0pt, after skip=0pt,
    fontupper=\small,
  }},
  imebubble/.style={{
    colback=IGBlue, colframe=IGBlue,
    coltext=white,
    arc=2.6mm, boxrule=0pt, left=2.2mm, right=2.2mm, top=1mm, bottom=1mm,
    width=\linewidth, before skip=0pt, after skip=0pt,
    fontupper=\small\color{{white}},
  }},
  stickerpanel/.style={{
    enhanced, colback=white, colframe=PhoneBorder,
    arc=3mm, boxrule=0.5pt, left=2.5mm, right=2.5mm, top=1.5mm, bottom=1.5mm,
    width=\linewidth,
  }},
}}

\begin{{document}}
{title_block}{body}
\end{{document}}
"""


def compile_tex(tex_path: Path, crop: bool = False) -> Path:
    if not PDFLATEX.exists():
        raise FileNotFoundError(f"pdflatex not found at {PDFLATEX}")
    for _ in range(2):
        subprocess.run(
            [str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    pdf_path = tex_path.with_suffix(".pdf")
    if crop and PDFCROP.exists():
        cropped = pdf_path.with_name(pdf_path.stem + "_cropped.pdf")
        subprocess.run(
            [str(PDFCROP), "--margins", "4 4 4 4", str(pdf_path), str(cropped)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cropped.replace(pdf_path)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="all", help="Case list/range, e.g. 1-5, 6-20, or all.")
    parser.add_argument("--combined", action="store_true", help="Also build one combined PDF.")
    parser.add_argument("--column", action="store_true", help="Render at CVPR column width (~3.3in).")
    args = parser.parse_args()
    suffix_tag = "_col" if args.column else ""

    selected = select_case_dirs(args.cases)
    if not selected:
        raise SystemExit(f"No cases matched {args.cases}")

    built = []
    for case_dir in selected:
        with open(case_dir / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        body = render_case_block(case_dir, metadata, case_dir, column_mode=args.column)
        tex_path = case_dir / f"{case_dir.name}_demo{suffix_tag}.tex"
        sample_id = metadata.get("sample_id", case_dir.name)
        title = f"Sticker Retrieval Demo — {sample_id}"
        tex_path.write_text(document(body, title, column=args.column), encoding="utf-8")
        built.append(compile_tex(tex_path, crop=False))

    if args.combined:
        blocks = []
        for case_dir in selected:
            with open(case_dir / "metadata.json", encoding="utf-8") as f:
                metadata = json.load(f)
            blocks.append(render_case_block(case_dir, metadata, DEMO_DIR, column_mode=args.column))
        suffix = args.cases.replace(",", "_").replace("-", "_")
        tex_path = DEMO_DIR / f"cases_{suffix}_demo{suffix_tag}.tex"
        tex_path.write_text(
            document("\\clearpage\n".join(blocks), f"Sticker Retrieval Demos {args.cases}", column=args.column),
            encoding="utf-8",
        )
        built.append(compile_tex(tex_path, crop=False))

    for pdf in built:
        print(pdf)


if __name__ == "__main__":
    main()
