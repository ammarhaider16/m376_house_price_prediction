import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ffnn.layers import get_layers_A, get_layers_B, get_layers_C, get_layers_D, get_layers_E


ARCH_FUNCTIONS = [get_layers_A, get_layers_B, get_layers_C, get_layers_D, get_layers_E]
OUT_PATH       = "figures/architectures.png"

# colours 
COL_INPUT  = "#C0DD97"
COL_HIDDEN = "#B5D4F4"
COL_OUTPUT = "#F5C4B3"
BRD_INPUT  = "#3B6D11"
BRD_HIDDEN = "#378ADD"
BRD_OUTPUT = "#993C1D"
TXT_INPUT  = "#27500A"
TXT_HIDDEN = "#0C447C"
TXT_OUTPUT = "#712B13"
COL_ARROW  = "#888780"
COL_ACT    = "#5F5E5A"
COL_TITLE  = "#2C2C2A"

# layout 
BOX_H  = 0.10
Y_TOP  = 0.90
Y_BOT  = 0.08
MAX_W  = 0.62
MIN_W  = 0.15


# helpers 
def parse_module_list(module_list):
    parsed = []
    for layer in module_list:
        name = type(layer).__name__
        if name == "Sequential":
            children = list(layer.children())
            linear = next((c for c in children if type(c).__name__ == "Linear"), None)
            acts   = [type(c).__name__ for c in children if type(c).__name__ != "Linear"]
            act    = acts[0] if acts else None
            if linear is not None:
                parsed.append({"in": linear.in_features, "out": linear.out_features, "act": act})
        elif name == "Linear":
            parsed.append({"in": layer.in_features, "out": layer.out_features, "act": None})
    return parsed


def layer_style(index, total):
    if index == 0:
        return COL_INPUT, BRD_INPUT, TXT_INPUT
    elif index == total - 1:
        return COL_OUTPUT, BRD_OUTPUT, TXT_OUTPUT
    else:
        return COL_HIDDEN, BRD_HIDDEN, TXT_HIDDEN

def draw_architecture(ax, layers, title, global_max_sz):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    n       = len(layers)
    total_h = Y_TOP - Y_BOT
    gap     = (total_h - n * BOX_H) / max(n - 1, 1)
    cx      = 0.5

    nodes = []
    first = layers[0]
    nodes.append({"size": first["in"], "label": "Input", "act": first["act"]})
    for l in layers:
        label = "Output" if l is layers[-1] else "Hidden"
        nodes.append({"size": l["out"], "label": label, "act": None})

    acts = [l["act"] for l in layers]

    n_nodes = len(nodes)
    gap     = (total_h - n_nodes * BOX_H) / max(n_nodes - 1, 1)

    for i, node in enumerate(nodes):
        y_top_edge = Y_TOP - i * (BOX_H + gap)
        y_ctr      = y_top_edge - BOX_H / 2

        # GLOBAL scaling
        w = MIN_W + (node["size"] / global_max_sz) * (MAX_W - MIN_W)

        x = cx - w / 2
        fc, ec, tc = layer_style(i, n_nodes)

        box = mpatches.FancyBboxPatch(
            (x, y_top_edge - BOX_H), w, BOX_H,
            boxstyle="round,pad=0.012",
            facecolor=fc, edgecolor=ec,
            linewidth=0.8, zorder=3,
        )
        ax.add_patch(box)

        ax.text(cx, y_ctr + BOX_H * 0.13, node["label"],
                ha="center", va="center",
                fontsize=9, fontweight="500", color=tc, zorder=4)

        ax.text(cx, y_ctr - BOX_H * 0.18, f'{node["size"]} units',
                ha="center", va="center",
                fontsize=7.5, color=tc, alpha=0.85, zorder=4)

        if i < n_nodes - 1:
            y_arrow_start = y_top_edge - BOX_H
            y_arrow_end   = y_top_edge - BOX_H - gap

            ax.annotate(
                "", xy=(cx, y_arrow_end + 0.006),
                xytext=(cx, y_arrow_start - 0.006),
                arrowprops=dict(arrowstyle="-|>", color=COL_ARROW,
                                lw=1.0, mutation_scale=9),
                zorder=2,
            )

            act = acts[i]
            arrow_label = act if act else "Linear"

            ax.text(cx + 0.04, (y_arrow_start + y_arrow_end) / 2, arrow_label,
                    ha="left", va="center",
                    fontsize=7.5, color=COL_ACT, style="italic", zorder=4)

    ax.set_title(title, fontsize=10, fontweight="500",
                 color=COL_TITLE, pad=2)

if __name__ == "__main__":
    parsed_archs = []
    global_max_sz = 0

    for fn in ARCH_FUNCTIONS:
        module_list = fn()
        layers = parse_module_list(module_list)
        parsed_archs.append((fn, layers))

        for l in layers:
            global_max_sz = max(global_max_sz, l["out"], l["in"])

    fig, axes = plt.subplots(
        1, len(parsed_archs),
        figsize=(3.2 * len(parsed_archs), 7)
    )
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0, hspace=0)   

    fig.patch.set_facecolor("white")

    if len(parsed_archs) == 1:
        axes = [axes]

    for ax, (fn, layers) in zip(axes, parsed_archs):
        name = fn.__name__.replace("get_layers_", "FFNN ")
        draw_architecture(ax, layers, name, global_max_sz)
  
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {OUT_PATH}")