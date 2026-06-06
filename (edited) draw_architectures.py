"""
Create publication-quality architecture diagrams for the 3 encoder-decoder models.
Output: 3 separate PNG files + 1 combined figure for the report.

Corrected to match the final notebook implementation:
- ResNet-34 blocks are named by torchvision modules: Conv1, Layer1, Layer2, Layer3, Layer4.
- U-Net decoder channels are aligned with the implemented ResNet-34 encoder-decoder.
- SegNet labels clarify that encoder sizes are after MaxPool and Decode1 is 64 channels before classifier.
- DeepLabV3+ uses ResNet-34 default output stride 32: ASPP 8x8 -> upsample x8 -> low-level fusion at 64x64.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ============================================================
# Style settings
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'figure.dpi': 300,
})

# Color palette
COLORS = {
    'encoder':   '#4A90D9',   # Blue
    'decoder':   '#E8834A',   # Orange
    'skip':      '#50C878',   # Green
    'bottleneck':'#9B59B6',   # Purple
    'input':     '#95A5A6',   # Gray
    'output':    '#E74C3C',   # Red
    'aspp':      '#F39C12',   # Yellow-orange
    'pool_idx':  '#1ABC9C',   # Teal
    'text_dark': '#2C3E50',
    'bg':        '#FAFAFA',
}


def draw_block(ax, x, y, w, h, label, color, fontsize=7, text_color='white'):
    """Draw a rounded rectangle block with label."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor='white',
                         linewidth=0.8, alpha=0.92)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color,
            family='serif')
    return (x, y, w, h)


def draw_arrow(ax, x1, y1, x2, y2, color='#555', style='->', lw=1.0, ls='-'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, linestyle=ls,
                                connectionstyle='arc3,rad=0'))


def draw_curved_arrow(ax, x1, y1, x2, y2, color='#555', rad=0.3, lw=1.0, label=''):
    """Draw a curved arrow."""
    ax.annotate(label, xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                fontsize=6, color=color, ha='center', va='bottom')


# ============================================================
# 1. U-Net + ResNet34 Architecture
# ============================================================
def draw_unet(ax):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) U-Net + ResNet-34', fontsize=10, fontweight='bold',
                 pad=8, family='serif')

    bw, bh = 1.75, 0.58  # block width, height

    # Input
    draw_block(ax, 1.0, 6.5, 1.4, 0.45, 'Input\n256×256×3', COLORS['input'], fontsize=6)

    # Encoder (left side - going down)
    # Correct naming follows torchvision ResNet-34 modules.
    enc_labels = [
        'Enc0: Conv1\n128×128×64',
        'Enc1: Layer1\n64×64×64',
        'Enc2: Layer2\n32×32×128',
        'Enc3: Layer3\n16×16×256',
    ]
    enc_y = [5.5, 4.3, 3.1, 1.9]
    for label, y in zip(enc_labels, enc_y):
        draw_block(ax, 1.0, y, bw, bh, label, COLORS['encoder'], fontsize=5.2)

    # Arrows encoder
    draw_arrow(ax, 1.0, 6.5-0.22, 1.0, 5.5+0.29, COLORS['encoder'])
    for i in range(len(enc_y)-1):
        draw_arrow(ax, 1.0, enc_y[i]-0.29, 1.0, enc_y[i+1]+0.29, COLORS['encoder'])

    # Bottleneck
    draw_block(ax, 5.0, 0.5, 2.2, 0.62,
               'Enc4 / Bottleneck\nLayer4\n8×8×512',
               COLORS['bottleneck'], fontsize=5.5)
    draw_arrow(ax, 1.0, 1.9-0.29, 1.0, 0.5, COLORS['encoder'])
    draw_arrow(ax, 1.9, 0.5, 3.9, 0.5, COLORS['bottleneck'])

    # Decoder (right side - going up)
    # Channels reflect the implemented U-Net decoder outputs before the next upsample.
    dec_labels = [
        'Up4\n16×16×256',
        'Up3\n32×32×128',
        'Up2\n64×64×64',
        'Up1\n128×128×64',
    ]
    dec_y = [1.9, 3.1, 4.3, 5.5]
    for label, y in zip(dec_labels, dec_y):
        draw_block(ax, 9.0, y, bw, bh, label, COLORS['decoder'], fontsize=5.4)

    # Arrows decoder
    draw_arrow(ax, 6.1, 0.5, 9.0, 1.9-0.29, COLORS['decoder'])
    for i in range(len(dec_y)-1):
        draw_arrow(ax, 9.0, dec_y[i]+0.29, 9.0, dec_y[i+1]-0.29, COLORS['decoder'])

    # Skip connections (horizontal arrows)
    # Enc3->Up4, Enc2->Up3, Enc1->Up2, Enc0->Up1.
    for i in range(4):
        enc_x_right = 1.0 + bw/2
        dec_x_left = 9.0 - bw/2
        y = enc_y[3-i]
        ax.annotate('', xy=(dec_x_left, y), xytext=(enc_x_right, y),
                    arrowprops=dict(arrowstyle='->', color=COLORS['skip'],
                                    lw=1.5, linestyle='--',
                                    connectionstyle='arc3,rad=0'))
        ax.text(5.0, y + 0.15, 'concat', ha='center', va='bottom',
                fontsize=5, color=COLORS['skip'], fontstyle='italic')

    # Output / final upsampling head
    draw_block(ax, 9.0, 6.5, 1.65, 0.5,
               'Up0 + Head\n256×256×C', COLORS['output'], fontsize=5.7)
    draw_arrow(ax, 9.0, 5.5+0.29, 9.0, 6.5-0.25, COLORS['decoder'])

    # Legend annotation
    ax.text(5.0, 7.2, 'Skip Connections (Concatenation)',
            ha='center', va='center', fontsize=6, color=COLORS['skip'],
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['skip'], alpha=0.8))


# ============================================================
# 2. SegNet + VGG16-BN Architecture
# ============================================================
def draw_segnet(ax):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(b) SegNet + VGG16-BN', fontsize=10, fontweight='bold',
                 pad=8, family='serif')

    bw, bh = 1.75, 0.58

    # Input
    draw_block(ax, 1.0, 6.5, 1.4, 0.45, 'Input\n256×256×3', COLORS['input'], fontsize=6)

    # Encoder blocks (VGG16 style).
    # Sizes shown are after MaxPool; only pooling indices are transferred.
    enc_labels = [
        'VGG Block1 + Pool\n128×128×64',
        'VGG Block2 + Pool\n64×64×128',
        'VGG Block3 + Pool\n32×32×256',
        'VGG Block4 + Pool\n16×16×512',
        'VGG Block5 + Pool\n8×8×512',
    ]
    enc_y = [5.5, 4.5, 3.5, 2.5, 1.5]
    for label, y in zip(enc_labels, enc_y):
        draw_block(ax, 1.0, y, bw, bh, label, COLORS['encoder'], fontsize=5.0)

    draw_arrow(ax, 1.0, 6.5-0.22, 1.0, 5.5+0.29, COLORS['encoder'])
    for i in range(len(enc_y)-1):
        draw_arrow(ax, 1.0, enc_y[i]-0.29, 1.0, enc_y[i+1]+0.29, COLORS['encoder'])
        ax.text(1.0 + bw/2 + 0.1, (enc_y[i] + enc_y[i+1])/2, 'MaxPool\n↓ indices',
                ha='left', va='center', fontsize=4.5, color=COLORS['pool_idx'])

    # Decoder blocks. Decode1 remains 64 channels; the final classifier maps to C classes.
    dec_labels = [
        'Decode5\n16×16×512',
        'Decode4\n32×32×256',
        'Decode3\n64×64×128',
        'Decode2\n128×128×64',
        'Decode1\n256×256×64',
    ]
    dec_y = [1.5, 2.5, 3.5, 4.5, 5.5]
    for label, y in zip(dec_labels, dec_y):
        draw_block(ax, 9.0, y, bw, bh, label, COLORS['decoder'], fontsize=5.4)

    for i in range(len(dec_y)-1):
        draw_arrow(ax, 9.0, dec_y[i]+0.29, 9.0, dec_y[i+1]-0.29, COLORS['decoder'])
        ax.text(9.0 - bw/2 - 0.1, (dec_y[i] + dec_y[i+1])/2, 'MaxUnpool\n↑',
                ha='right', va='center', fontsize=4.5, color=COLORS['pool_idx'])

    # Pooling indices transfer (curved dotted arrows, not feature concatenation)
    for i in range(5):
        enc_x = 1.0 + bw/2
        dec_x = 9.0 - bw/2
        y_enc = enc_y[i]
        y_dec = dec_y[4-i]
        ax.annotate('', xy=(dec_x, y_dec), xytext=(enc_x, y_enc),
                    arrowprops=dict(arrowstyle='->', color=COLORS['pool_idx'],
                                    lw=1.2, linestyle=':',
                                    connectionstyle='arc3,rad=-0.15'))

    # Bottom connection from deepest encoder features to first decoder block
    draw_arrow(ax, 1.9, 1.5, 8.1, 1.5, COLORS['bottleneck'], lw=1.5)

    # Output classifier
    draw_block(ax, 9.0, 6.5, 1.55, 0.5,
               'Classifier 1×1\n256×256×C', COLORS['output'], fontsize=5.6)
    draw_arrow(ax, 9.0, 5.5+0.29, 9.0, 6.5-0.25, COLORS['decoder'])

    # Legend
    ax.text(5.0, 7.2, 'Pooling Indices Transfer (No feature skip)',
            ha='center', va='center', fontsize=6, color=COLORS['pool_idx'],
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['pool_idx'], alpha=0.8))


# ============================================================
# 3. DeepLabV3+ + ResNet34 Architecture
# ============================================================
def draw_deeplabv3plus(ax):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(c) DeepLabV3+ + ResNet-34', fontsize=10, fontweight='bold',
                 pad=8, family='serif')

    bw, bh = 1.75, 0.58

    # Input
    draw_block(ax, 0.8, 6.5, 1.4, 0.45, 'Input\n256×256×3', COLORS['input'], fontsize=6)

    # Encoder. Names follow torchvision ResNet-34 modules.
    enc_labels = [
        'Layer1 / Low-level\n64×64×64',
        'Layer2\n32×32×128',
        'Layer3\n16×16×256',
        'Layer4 / High-level\n8×8×512',
    ]
    enc_y = [5.3, 4.1, 2.9, 1.7]
    for label, y in zip(enc_labels, enc_y):
        draw_block(ax, 0.8, y, bw, bh, label, COLORS['encoder'], fontsize=5.0)

    draw_arrow(ax, 0.8, 6.5-0.22, 0.8, 5.3+0.29, COLORS['encoder'])
    for i in range(len(enc_y)-1):
        draw_arrow(ax, 0.8, enc_y[i]-0.29, 0.8, enc_y[i+1]+0.29, COLORS['encoder'])

    # ASPP Module (center) receives high-level 8x8 features.
    aspp_x, aspp_y = 5.0, 1.7
    draw_block(ax, aspp_x, aspp_y, 2.8, 0.72,
               'ASPP Module\n8×8×256', COLORS['aspp'],
               fontsize=6.2, text_color='white')

    # ASPP branches
    branches = ['1×1\nconv', 'rate\n6', 'rate\n12', 'rate\n18', 'Global\nAvgPool']
    branch_x = [3.8, 4.4, 5.0, 5.6, 6.2]
    for label, bx in zip(branches, branch_x):
        draw_block(ax, bx, 0.5, 0.5, 0.5, label, '#E8C547', fontsize=4, text_color='#333')

    # Arrow from high-level encoder feature to ASPP
    draw_arrow(ax, 0.8 + bw/2, 1.7, aspp_x - 1.4, 1.7, COLORS['encoder'], lw=1.5)

    # Arrows from ASPP to branches and back to ASPP projection
    for bx in branch_x:
        draw_arrow(ax, bx, aspp_y - 0.36, bx, 0.75, '#E8C547', lw=0.7)

    # ASPP branch concatenation and projection, still at 8x8 resolution
    draw_block(ax, 5.0, -0.3, 2.25, 0.48,
               'ASPP concat + 1×1 proj\n8×8×256',
               COLORS['bottleneck'], fontsize=4.8)
    for bx in branch_x:
        draw_arrow(ax, bx, 0.25, 5.0, -0.08, '#999', lw=0.5)

    # Decoder path
    draw_block(ax, 8.5, 2.0, 1.9, 0.6,
               'Upsample ×8\n64×64×256', COLORS['decoder'], fontsize=5.4)
    draw_block(ax, 8.5, 3.5, 1.95, 0.72,
               'Concat 304ch\n+ 3×3 Conv×2\n64×64×C', COLORS['decoder'], fontsize=4.8)
    draw_block(ax, 8.5, 5.0, 1.9, 0.58,
               'Final Upsample ×4\n256×256×C', COLORS['decoder'], fontsize=5.1)

    # Arrow: ASPP projection -> upsample to low-level spatial size
    draw_arrow(ax, 6.1, -0.3, 8.5, 1.7, COLORS['decoder'], lw=1.2)
    ax.text(7.2, 0.85, 'bilinear\nupsample', ha='center', va='center',
            fontsize=4.5, color=COLORS['decoder'], fontstyle='italic')

    # Arrow: decoder blocks
    draw_arrow(ax, 8.5, 2.3, 8.5, 3.12, COLORS['decoder'])
    draw_arrow(ax, 8.5, 3.88, 8.5, 4.71, COLORS['decoder'])

    # Low-level feature branch: Layer1 -> 1x1 projection -> concat
    draw_block(ax, 6.45, 3.5, 1.55, 0.62,
               'Low-level\n1×1 Conv\n64×64×48', COLORS['skip'], fontsize=4.6)
    ax.annotate('', xy=(5.68, 3.5), xytext=(0.8 + bw/2, 5.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['skip'],
                                lw=1.8, linestyle='--',
                                connectionstyle='arc3,rad=0.28'))
    draw_arrow(ax, 7.23, 3.5, 7.53, 3.5, COLORS['skip'], lw=1.2)
    ax.text(4.2, 4.85, 'Low-level features', ha='center', va='center',
            fontsize=5.0, color=COLORS['skip'], fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=COLORS['skip'], alpha=0.7))

    # Output
    draw_block(ax, 8.5, 6.2, 1.4, 0.45, 'Output\n256×256×C', COLORS['output'], fontsize=6)
    draw_arrow(ax, 8.5, 5.29, 8.5, 6.2-0.22, COLORS['decoder'])

    # Legend
    ax.text(5.0, 7.2, 'ASPP Multi-scale + Low-level Fusion',
            ha='center', va='center', fontsize=6, color=COLORS['aspp'],
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLORS['aspp'], alpha=0.8))


# ============================================================
# Generate all diagrams
# ============================================================

def main():
    out_dir = Path('outputs_topic05_3')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined figure (for report - fits in 1 column or 2 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('white')

    draw_unet(axes[0])
    draw_segnet(axes[1])
    draw_deeplabv3plus(axes[2])

    plt.tight_layout(pad=1.5)
    combined_path = out_dir / 'architecture_diagrams_combined.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {combined_path}")
    plt.close()

    # Individual figures
    for name, draw_func in [('unet', draw_unet),
                            ('segnet', draw_segnet),
                            ('deeplabv3plus', draw_deeplabv3plus)]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 7))
        fig.patch.set_facecolor('white')
        draw_func(ax)
        plt.tight_layout(pad=1.0)
        path = out_dir / f'architecture_{name}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {path}")
        plt.close()

    print("\nDone! All architecture diagrams saved.")


if __name__ == '__main__':
    main()
