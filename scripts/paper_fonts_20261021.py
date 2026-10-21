"""Manuscript font adjustment for Figs. 5 and 8-10 (2026-10-21, author request), opt-in with F1_PAPER_FONTS=1.

Every text element of the figure (titles, axis labels, annotations, legend) is scaled by SCALE, and the tick labels of
both axes take the same size as the legend (or REF_DEFAULT if the figure has no legend), so the axis numbers have the
same format as the other text. Call apply(fig) right before saving. Without the variable it does nothing.
"""
import os

SCALE = 0.9
REF_DEFAULT = 9.0


def enabled():
    return bool(os.environ.get('F1_PAPER_FONTS'))


def apply(fig):
    if not enabled():
        return
    fig.canvas.draw()
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    sizes = [t.get_fontsize() for lg in legends for t in lg.get_texts()]
    ref = (max(set(sizes), key=sizes.count) if sizes else REF_DEFAULT) * SCALE
    ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            ticks.update(id(t) for t in axis.get_ticklabels(which='both'))
    for t in fig.findobj(lambda a: hasattr(a, 'set_fontsize') and hasattr(a, 'get_text')):
        if id(t) not in ticks:
            t.set_fontsize(t.get_fontsize() * SCALE)
    for ax in fig.axes:
        ax.tick_params(axis='both', which='both', labelsize=ref)
    print(f'[paper_fonts] scale {SCALE}, tick labels {ref:.2f} pt', flush=True)
