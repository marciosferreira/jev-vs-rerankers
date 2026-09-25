"""Render the article's tables as PNG images (Medium does not render Markdown tables).

Usage: python make_figures.py   ->   figures/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#d9d8d3"
ACCENT = "#2a78d6"  # highlighted column (JEV)
ACCENT_TINT = "#e8f0fb"

FONT = "DejaVu Sans"
SIZE = 12
ROW_H = 0.42  # inches
PAD_X = 0.14  # inches of horizontal padding per cell


def text_width(s: str, bold: bool = False) -> float:
    """Rough width in inches for layout (DejaVu Sans at SIZE pt)."""
    return len(s) * SIZE / 72 * (0.62 if bold else 0.58)


def table(name: str, title: str, header: list[str], rows: list[list[str]],
          highlight: int | None = None, bold: set[tuple[int, int]] = frozenset(),
          note: str = "", first_col_min: float = 0.0, left_cols: set[int] = frozenset({0})) -> None:
    """bold: (row, col) cells to emphasize; highlight: column index to tint;
    left_cols: text columns aligned left (numbers are centered)."""
    ncol = len(header)
    widths = []
    for c in range(ncol):
        cells = [header[c]] + [r[c] for r in rows]
        w = max(text_width(s, bold=True) for s in cells) + 2 * PAD_X
        widths.append(max(w, first_col_min) if c == 0 else max(w, 1.0))
    width = sum(widths) + 0.3
    height = ROW_H * (len(rows) + 1) + 0.75 + (0.35 * (note.count("\n") + 1) if note else 0.25)

    fig = plt.figure(figsize=(width, height), dpi=220)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # y grows downward
    ax.axis("off")

    x0, y = 0.15, 0.2
    ax.text(x0, y, title, fontsize=SIZE + 3, fontweight="bold", color=INK, va="top", family=FONT)
    y += 0.55

    xs = [x0]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    table_top = y

    if highlight is not None:
        ax.add_patch(plt.Rectangle((xs[highlight], table_top), widths[highlight],
                                   ROW_H * (len(rows) + 1), color=ACCENT_TINT, lw=0))

    def cell(r: int, c: int, s: str, is_header: bool) -> None:
        left = c in left_cols
        cx = xs[c] + (PAD_X if left else widths[c] / 2)
        ha = "left" if left else "center"
        weight = "bold" if is_header or (r, c) in bold else "normal"
        color = ACCENT if is_header and c == highlight else (INK if is_header or c == 0 or (r, c) in bold else INK)
        if not is_header and c == 0:
            color = INK_2
        ax.text(cx, y + ROW_H / 2, s, fontsize=SIZE, fontweight=weight, color=color,
                ha=ha, va="center", family=FONT)

    for c, h in enumerate(header):
        cell(-1, c, h, True)
    y += ROW_H
    ax.plot([x0, x0 + sum(widths)], [y, y], color=INK, lw=1.0)
    for r, row in enumerate(rows):
        for c, s in enumerate(row):
            cell(r, c, s, False)
        y += ROW_H
        ax.plot([x0, x0 + sum(widths)], [y, y], color=RULE, lw=0.6)

    if note:
        ax.text(x0, y + 0.18, note, fontsize=SIZE - 2, color=INK_2, va="top", family=FONT, linespacing=1.4)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"figures/{name}.png")


def best(rows: list[list[str]], r: int, cols: list[int], lower: bool) -> set[tuple[int, int]]:
    """Cells holding the best numeric value of row r among cols."""
    def num(s: str) -> float:
        return float(s.split()[0].replace("%", "").replace("$", "").replace(",", ""))

    vals = {c: num(rows[r][c]) for c in cols}
    target = min(vals.values()) if lower else max(vals.values())
    return {(r, c) for c, v in vals.items() if v == target}


def main() -> None:
    # 1. the four context policies
    table(
        "fig1_policies",
        "The four context policies",
        ["Policy", "What reaches the LLM"],
        [
            ["top3", "the 3 passages most similar by embedding (common baseline)"],
            ["cohere_top3", "the 3 best passages by Cohere Rerank 4 Pro (common \"serious\" setup)"],
            ["cohere_cut", "every passage with Cohere score ≥ 0.891 (less common setup)"],
            ["jev_cut", "every passage with JEV probability ≥ 0.97; no LLM call if none passes"],
        ],
        first_col_min=1.6, left_cols={0, 1},
    )

    # 2. main end-to-end results
    head = ["299 SQuAD 2.0 questions", "top3", "cohere_top3", "cohere_cut", "jev_cut"]
    rows = [
        ["Hallucinated answers", "24 (8.0%)", "23 (7.7%)", "7 (2.3%)", "3 (1.0%)"],
        ["Confidently wrong answers", "46 (15.4%)", "48 (16.1%)", "24 (8.0%)", "13 (4.3%)"],
        ["…on the 54 unanswerable questions", "35 (65%)", "37 (69%)", "17 (31%)", "8 (15%)"],
        ["Correct answers, all", "81.3%", "81.9%", "79.3%", "83.6%"],
        ["Correct, answerable questions", "91%", "94%", "82%", "83%"],
        ["\"I don't know\" when an answer existed", "4%", "2%", "15%", "15%"],
        ["Questions answered without calling the LLM", "0%", "0%", "22%", "27%"],
    ]
    cols = [1, 2, 3, 4]
    b = set()
    for r, lower in enumerate([True, True, True, False, False, True]):  # last row (no LLM call) is cost, not quality
        b |= best(rows, r, cols, lower)
    table(
        "fig2_results",
        "Wrong answers and accuracy by context policy",
        head, rows, highlight=4, bold=b, first_col_min=3.9,
        note="Bold: best value in each quality row. Judge: claude-sonnet-5. Generator: gpt-4.1-mini. One run.\n"
             "Every answer is exactly one of: correct, confidently wrong (includes hallucinations), or an\n"
             "incorrect \"I don't know\" (a false abstention, or an abstention mixed with an unsupported claim).",
    )

    # 3. cost
    head = ["Per 1,000 questions", "top3", "cohere_top3", "cohere_cut", "jev_cut"]
    rows = [
        ["LLM (measured)", "$0.341", "$0.341", "$0.150", "$0.133"],
        ["Passage scoring", "–", "$2.50*", "$2.50*", "$0.256"],
        ["Total", "$0.341", "$2.841", "$2.650", "$0.389"],
    ]
    b = best(rows, 0, cols, True) | {(1, 4)} | best(rows, 2, cols, True)
    table(
        "fig3_cost",
        "Cost per 1,000 questions",
        head, rows, highlight=4, bold=b, first_col_min=2.4,
        note="Bold: lowest cost in each row. * Cohere list price (~$2.50 per 1,000 searches).\n"
             "JEV and LLM costs measured from OpenRouter usage. The JEV cost includes an 11th call\n"
             "per question that jev_cut doesn't use, so it is conservative (~9% high).",
    )

    # 4. ordering vs deciding
    head = ["", "Embedding", "bge-reranker-v2-m3", "Cohere Rerank 4 Pro", "JEV"]
    rows = [
        ["Top-1 passage answers", "85%", "97%", "97%", "95%"],
        ["Detects \"no useful passage\" (AUC)", "0.69", "0.79", "0.85", "0.95"],
        ["Withholds useless context*", "31%", "52%", "61%", "87%"],
        ["…used as top-k (the usual way)", "0%", "0%", "0%", "–"],
    ]
    cols = [1, 2, 3, 4]
    b = best(rows, 0, cols, False) | best(rows, 1, cols, False) | best(rows, 2, cols, False)
    table(
        "fig4_ordering_vs_deciding",
        "Rerankers order better. JEV decides better.",
        head, rows, highlight=4, bold=b, first_col_min=3.6,
        note="Retrieval only, no LLM: all 300 test questions, 10 candidates each. Ordering: the 239\n"
             "with a useful candidate. Deciding: the 61 without one. * With each method's cut-off set so that\n"
             "85% of questions get a useful passage. Bold: best in row. MiniLM and BGE-base in the repository.",
    )


if __name__ == "__main__":
    main()
