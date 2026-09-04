"""Draws a live diagram of a NEAT genome's network: nodes laid out
input -> hidden -> output (using NEAT's own feed-forward layering),
edges colored by weight sign (green = positive, red = negative) and
faded by magnitude (weak connections fade into the background so
strong ones read clearly), with each node's current activation value
pulled from net.values (populated by FeedForwardNetwork after
net.activate() runs for that car this frame)."""

import pygame
import pygame.gfxdraw
from neat.graphs import feed_forward_layers

NODE_RADIUS = 13
INPUT_COLOR = (86, 168, 224)
HIDDEN_COLOR = (222, 196, 86)
OUTPUT_COLOR = (235, 235, 238)
NODE_OUTLINE = (12, 13, 16)
POS_EDGE = (96, 200, 118)
NEG_EDGE = (214, 82, 78)
LABEL_COLOR = (198, 200, 206)
VALUE_COLOR = (18, 18, 20)
PANEL_BG_TOP = (16, 17, 21)
PANEL_BG_BOTTOM = (24, 26, 31)
DIVIDER_COLOR = (70, 74, 82)
HEADER_COLOR = (140, 144, 152)

MIN_EDGE_ALPHA = 35
MAX_EDGE_ALPHA = 220
MIN_EDGE_WIDTH = 1
MAX_EDGE_WIDTH = 4
PADDING_X = 70
PADDING_Y = 34

_font_cache = {}


def _font(size=13, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = pygame.font.SysFont("Arial", size, bold=bold)
    return _font_cache[key]


def _resolve_layers(input_keys, output_keys, connections):
    result = feed_forward_layers(input_keys, output_keys, connections)
    return result[0] if isinstance(result, tuple) else result  # newer neat-python returns (layers, required)


def _node_positions(config, genome, w, h):
    """Returns node positions in LOCAL panel coordinates (0..w, 0..h)."""
    gc = config.genome_config
    input_keys = list(gc.input_keys)

    enabled_conns = [cg.key for cg in genome.connections.values() if cg.enabled]
    layers = _resolve_layers(input_keys, gc.output_keys, enabled_conns)
    columns = [input_keys] + [sorted(layer) for layer in layers if layer]

    usable_w = max(w - 2 * PADDING_X, 1)
    usable_h = max(h - 2 * PADDING_Y, 1)

    positions = {}
    n_cols = max(len(columns), 1)
    for ci, col_nodes in enumerate(columns):
        x = PADDING_X + (0 if n_cols == 1 else ci * (usable_w / (n_cols - 1)))
        n = max(len(col_nodes), 1)
        for ni, node in enumerate(col_nodes):
            y = PADDING_Y + (ni + 0.5) * (usable_h / n)
            positions[node] = (x, y)
    return positions


def _draw_background(screen, x0, y0, w, h):
    panel = pygame.Surface((w, h))
    for row in range(h):
        t = row / max(h - 1, 1)
        color = tuple(
            int(PANEL_BG_TOP[i] + (PANEL_BG_BOTTOM[i] - PANEL_BG_TOP[i]) * t)
            for i in range(3)
        )
        pygame.draw.line(panel, color, (0, row), (w, row))
    screen.blit(panel, (x0, y0))
    pygame.draw.line(screen, DIVIDER_COLOR, (x0, y0), (x0 + w, y0), 2)


def _draw_edges(screen, x0, y0, w, h, genome, positions):
    weights = [abs(cg.weight) for cg in genome.connections.values() if cg.enabled]
    max_weight = max(weights, default=1.0) or 1.0

    edge_surface = pygame.Surface((w, h), pygame.SRCALPHA)
    for cg in genome.connections.values():
        if not cg.enabled:
            continue
        a, b = cg.key
        if a not in positions or b not in positions:
            continue

        strength = min(abs(cg.weight) / max_weight, 1.0)
        base_color = POS_EDGE if cg.weight >= 0 else NEG_EDGE
        alpha = int(MIN_EDGE_ALPHA + (MAX_EDGE_ALPHA - MIN_EDGE_ALPHA) * strength)
        width = max(MIN_EDGE_WIDTH, round(MIN_EDGE_WIDTH + (MAX_EDGE_WIDTH - MIN_EDGE_WIDTH) * strength))

        pygame.draw.line(edge_surface, (*base_color, alpha), positions[a], positions[b], width)

    screen.blit(edge_surface, (x0, y0))


def _draw_node(screen, x, y, color, value):
    pygame.gfxdraw.filled_circle(screen, int(x), int(y), NODE_RADIUS, color)
    pygame.gfxdraw.aacircle(screen, int(x), int(y), NODE_RADIUS, NODE_OUTLINE)
    pygame.gfxdraw.aacircle(screen, int(x), int(y), NODE_RADIUS, color)

    value_font = _font(11, bold=True)
    value_text = value_font.render(f"{value:.2f}", True, VALUE_COLOR)
    screen.blit(value_text, (x - value_text.get_width() / 2, y - value_text.get_height() / 2))


def draw_network(screen, rect, genome, config, net, input_labels=None, output_labels=None, title=None):
    x0, y0, w, h = rect
    _draw_background(screen, x0, y0, w, h)

    positions = _node_positions(config, genome, w, h)
    _draw_edges(screen, x0, y0, w, h, genome, positions)

    gc = config.genome_config
    input_keys, output_keys = list(gc.input_keys), list(gc.output_keys)
    label_font = _font(13)

    for node, (lx, ly) in positions.items():
        x, y = x0 + lx, y0 + ly

        if node in input_keys:
            color = INPUT_COLOR
        elif node in output_keys:
            color = OUTPUT_COLOR
        else:
            color = HIDDEN_COLOR

        value = net.values.get(node, 0.0)
        _draw_node(screen, x, y, color, value)

        if node in input_keys and input_labels:
            idx = input_keys.index(node)
            if idx < len(input_labels):
                label = label_font.render(input_labels[idx], True, LABEL_COLOR)
                screen.blit(label, (x - NODE_RADIUS - label.get_width() - 10, y - label.get_height() / 2))

        if node in output_keys and output_labels:
            idx = output_keys.index(node)
            if idx < len(output_labels):
                label = label_font.render(output_labels[idx], True, LABEL_COLOR)
                screen.blit(label, (x + NODE_RADIUS + 10, y - label.get_height() / 2))

    if title:
        header_font = _font(13, bold=True)
        header = header_font.render(title, True, HEADER_COLOR)
        screen.blit(header, (x0 + 14, y0 + 10))