from __future__ import annotations

import matplotlib.pyplot as plt


LETTER_PAGE_WIDTH_IN = 8.5
LETTER_PAGE_MARGIN_IN = 0.45
ROW_GAP_IN = 0.08
ROW_TOTAL_WIDTH_IN = LETTER_PAGE_WIDTH_IN - 2 * LETTER_PAGE_MARGIN_IN
HALF_PAGE_WIDTH_IN = (ROW_TOTAL_WIDTH_IN - ROW_GAP_IN) / 2
COMPARISON_WIDTH_IN = HALF_PAGE_WIDTH_IN
COMPARISON_HEIGHT_IN = 2.0
INTERPOLATION_WIDTH_IN = HALF_PAGE_WIDTH_IN
INTERPOLATION_HEIGHT_IN = 2.0


def apply_latex_plot_style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.02,
        }
    )
