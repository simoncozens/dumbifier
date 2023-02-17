from enum import IntEnum
import math
import logging
from fontTools.misc.transform import Transform
from fontTools.misc.roundTools import otRound
from fontTools.misc.bezierTools import (
    _alignment_transformation,
    calcCubicParameters,
    solveCubic,
    cubicPointAtT,
    linePointAtT,
    segmentPointAtT,
    splitCubic,
)
from glyphsLib.classes import GSAnchor
from glyphsLib.types import Point


logger = logging.getLogger(__name__)
null_anchor = GSAnchor()

ORIGIN = Point(0,0)

try:
    from math import dist
except ImportError:

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)



def otRoundNode(node):
    node.position.x, node.position.y = otRound(node.position.x), otRound(node.position.y)
    return node


# We often have GSNodes, but fontTools math stuff needs tuples.
def as_tuples(pts):
    return [(pt.position.x, pt.position.y) for pt in pts]


def get_next_segment(path, index):
    seg = [path.nodes[index]]
    index = (index + 1) % len(path.nodes)
    seg.append(path.nodes[index])
    if seg[-1].type == "offcurve":
        index = (index + 1) % len(path.nodes)
        seg.append(path.nodes[index])
        index = (index + 1) % len(path.nodes)
        seg.append(path.nodes[index])
    return seg


def get_previous_segment(path, index):
    seg = [path.nodes[index]]
    index = (index - 1) % len(path.nodes)
    seg.append(path.nodes[index])
    if seg[-1].type == "offcurve":
        index = (index - 1) % len(path.nodes)
        seg.append(path.nodes[index])
        index = (index - 1) % len(path.nodes)
        seg.append(path.nodes[index])
    return list(reversed(seg))


def closest_point_on_segment(seg, pt):
    # Everything here is a tuple
    if len(seg) == 2:
        return closest_point_on_line(seg, pt)
    return closest_point_on_cubic(seg, pt)


def closest_point_on_line(seg, pt):
    a, b = seg
    a_to_b = (b[0] - a[0], b[1] - a[1])
    a_to_pt = (pt[0] - a[0], pt[1] - a[1])
    mag = a_to_b[0] ** 2 + a_to_b[1] ** 2
    if mag == 0:
        return a
    atp_dot_atb = a_to_pt[0] * a_to_b[0] + a_to_pt[1] * a_to_b[1]
    t = atp_dot_atb / mag
    return (a[0] + a_to_b[0] * t, a[1] + a_to_b[1] * t)


def closest_point_on_cubic(bez, pt, start=0.0, end=1.0, iterations=5, slices=5):
    tick = (end - start) / slices
    best = 0
    best_dist = float("inf")
    t = start
    best_pt = pt
    while t < end:
        this_pt = cubicPointAtT(*bez, t)
        current_distance = dist(this_pt, pt)
        if current_distance <= best_dist:
            best_dist = current_distance
            best = t
            best_pt = this_pt
        t += tick
    if iterations < 1:
        return best_pt
    return closest_point_on_cubic(
        bez,
        pt,
        start=max(best - tick, 0),
        end=min(best + tick, 1),
        iterations=iterations - 1,
        slices=slices,
    )


def unbounded_seg_seg_intersection(seg1, seg2):
    if len(seg1) == 2 and len(seg2) == 2:
        aligned_seg1 = _alignment_transformation(seg1).transformPoints(seg2)
        if not math.isclose(aligned_seg1[0][1], aligned_seg1[1][1]):
            t = aligned_seg1[0][1] / (aligned_seg1[0][1] - aligned_seg1[1][1])
            return linePointAtT(*seg2, t)
        elif not math.isclose(aligned_seg1[0][0], aligned_seg1[1][0]):
            t = aligned_seg1[0][0] / (aligned_seg1[0][0] - aligned_seg1[1][0])
            return linePointAtT(*seg2, t)
        else:
            return None
    if len(seg1) == 4 and len(seg2) == 2:
        curve, line = seg1, seg2
    elif len(seg1) == 2 and len(seg2) == 4:
        line, curve = seg1, seg2
    aligned_curve = _alignment_transformation(line).transformPoints(curve)
    a, b, c, d = calcCubicParameters(*aligned_curve)
    intersections = solveCubic(a[1], b[1], c[1], d[1])
    real_intersections = [t for t in intersections if t >= 0 and t <= 1]
    if real_intersections:
        return cubicPointAtT(*curve, real_intersections[0])
    return None  # Needs bending


def point_on_seg_at_distance(seg, distance):
    aligned_seg = _alignment_transformation(seg).transformPoints(seg)
    if len(aligned_seg) == 4:
        a, b, c, d = calcCubicParameters(*aligned_seg)
        solutions = solveCubic(a[0], b[0], c[0], d[0] - (aligned_seg[0][0] + distance))
        solutions = sorted(t for t in solutions if 0 <= t < 1)
        if not solutions:
            return None
        return solutions[0]
    else:
        start, end = aligned_seg
        if math.isclose(end[0], start[0]):
            return distance / (end[1] - start[1])
        else:
            return distance / (end[0] - start[0])


def split_cubic_at_point(seg, point, inward=True):
    # There's a horrible edge case here where the curve wraps around and
    # the ray hits twice, but I'm not worrying about it.
    if inward:
        new_cubic_1 = splitCubic(*seg, point[0], False)[0]
        new_cubic_2 = splitCubic(*seg, point[1], True)[0]
    else:
        new_cubic_1 = splitCubic(*seg, point[0], False)[-1]
        new_cubic_2 = splitCubic(*seg, point[1], True)[-1]
    if dist(new_cubic_1[-1], point) < dist(new_cubic_2[-1], point):
        return new_cubic_1
    else:
        return new_cubic_2

class Alignment(IntEnum):
    OUTSTROKE = 0  # Glyphs calls this "left" alignment
    INSTROKE = 1  # Glyphs calls this "right" alignment
    MIDDLE = 2
    UNUSED = 3
    UNALIGNED = 4


class CornerComponentApplier:
    def __init__(self, hint, glyph, layer):
        self.glyph = glyph
        self.layer = layer

        self.corner_name = hint.name
        self.glyph_name = glyph.name
        self.alignment = Alignment(hint.options or 0)
        self.path_index, self.node_index = hint.origin

        self.scale = hint.scale
        if hint.name not in glyph.parent.glyphs:
            raise ValueError(f"Could not find corner component {hint.name} used in layer {layer}")
        corner_glyph = glyph.parent.glyphs[hint.name]
        matching_layers = [
            corner_layer
            for corner_layer in corner_glyph.layers
            if corner_layer.layerId == layer.layerId
                or corner_layer.associatedMasterId == layer.layerId
        ]
        if not matching_layers:
            raise ValueError(f"Could not find matching layer for corner component {hint.name} in layer {layer}")
        matching_layer = matching_layers[0]

        # Sadly we can't use .get here because the LayerAnchorsProxy hasn't got one
        self.left = (matching_layer.anchors["left"] or null_anchor).position
        self.right = (matching_layer.anchors["right"] or null_anchor).position
        self.origin = (matching_layer.anchors["origin"] or null_anchor).position
    
        # These need to be cloned as we will transform/scale/etc. them
        self.corner_path = matching_layer.paths[0].clone()
        self.other_paths = [path.clone() for path in matching_layer.paths[1:]]
        # This position may change later
        path = layer.paths[self.path_index]
        self.target_node = path.nodes[self.node_index]


    def fail(self, msg, hard=True):
        full_msg = f"{msg} (corner {self.corner_name} in {self.glyph_name})"
        if hard:
            raise ValueError(full_msg)
        else:
            logger.error(full_msg)

    @property
    def instroke(self):
        return get_previous_segment(self.path, self.target_node_ix)

    @property
    def outstroke(self):
        return get_next_segment(self.path, self.target_node_ix)

    @property
    def first_seg(self):
        return get_next_segment(self.corner_path, 0)

    @property
    def last_seg(self):
        return get_previous_segment(self.corner_path, len(self.corner_path.nodes) - 1)

    def apply(self):
        self.path = self.layer.paths[self.path_index]
        # Find where the target node lines in this path. This may have
        # changed, if we have applied a corner component in this path
        # already.
        for ix, node in enumerate(self.path.nodes):
            if node == self.target_node:
                self.target_node_ix = ix
        if self.target_node_ix is None:
            self.fail("Lost track of where the corner should be applied")

        if self.corner_path.nodes[0].position.x != self.origin[0]:
            self.fail(
                "Can't deal with offset instrokes yet; start corner components on axis",
                hard=False,
            )

        # This is for handling the left and right anchors and doesn't
        # quite work yet
        self.determine_start_and_end_vectors()

        # Align all paths to the "origin" anchor.
        for path in [self.corner_path] + self.other_paths:
            for node in path.nodes:
                node.position.x, node.position.y = node.position.x - self.origin[0], node.position.y - self.origin[1]

        # Apply scaling. We are considered "flipped" if one or other
        # of the scale factors is negative, but not both. Being flipped
        # means that the corner path gets applied backwards.
        self.flipped = False
        if self.scale is not None:
            self.flipped = (self.scale[0] * self.scale[1]) < 0
            self.scale_paths()

        # Align and rotate the corner paths so that they fit onto
        # the host path
        (
            instroke_intersection_point,
            outstroke_intersection_point,
        ) = self.align_my_path_to_main_path()

        # Keep hold of the original outstroke segment. Fitting the
        # instroke to the corner component will change the position
        # of the target node (since it's at the end of that segment)
        # so we need to recover it later.
        original_outstroke = as_tuples(self.outstroke)

        # If we are not aligned to the instroke, we need to re-fit the
        # instroke based on where we put the corner component, and
        # potentially stretch the corner component so that it meets the
        # instroke.
        if self.alignment != Alignment.INSTROKE:
            instroke_intersection_point = self.recompute_instroke_intersection_point()
            # The instroke of the corner path may need stretching to fit...
            if len(self.first_seg) == 4:
                self.stretch_first_seg_to_fit(instroke_intersection_point)
        self.split_instroke(instroke_intersection_point)

        # Now we insert the aligned and rotated corner path into the host
        nodelist = list(self.path.nodes)
        nodelist[self.target_node_ix + 1 : self.target_node_ix + 1] = [
            otRoundNode(node) for node in self.corner_path.nodes[1:]
        ]
        self.path.nodes = nodelist

        # And fix up the outstroke
        outstroke_intersection_point = self.recompute_outstroke_intersection_point(
            original_outstroke
        )
        self.fixup_outstroke(original_outstroke, outstroke_intersection_point)

        # Last of all, if there are other paths in the corner component,
        # they just get copied into the glyph.
        self.insert_other_paths()

    def determine_start_and_end_vectors(self):
        # Left and right anchors provide an additional offset, depending
        # on their relationship with the start/end of the first/last points
        # of the corner seg
        if self.left == ORIGIN:
            self.effective_start = (0, 0)
        else:
            self.effective_start = (
                self.corner_path.nodes[0].position.x - self.left.x,
                self.corner_path.nodes[0].position.y - self.left.y,
            )
            self.fail(
                "left and right anchors to corner components are"
                " not currently supported",
                hard=False,
            )

        if self.right == ORIGIN:
            self.effective_end = (0, 0)
        else:
            self.effective_end = (
                self.corner_path.nodes[-1].position.x - self.right.x,
                self.corner_path.nodes[-1].position.y - self.right.y,
            )
            self.fail(
                "left and right anchors to corner components are"
                " not currently supported",
                hard=False,
            )

    def scale_paths(self):
        scaling = Transform().scale(*self.scale)
        for path in [self.corner_path] + self.other_paths:
            for node in path.nodes:
                node.position.x, node.position.y = scaling.transformPoint((node.position.x, node.position.y))

    def align_my_path_to_main_path(self):
        # Work out my rotation (1): Rotation occurring due to corner paths
        angle = math.atan2(-self.corner_path.nodes[-1].position.y, self.corner_path.nodes[-1].position.x)

        # Work out my rotation (2): Rotation occurring due to host paths
        if self.flipped:
            angle += math.radians(90)

            self.reverse_corner_path()

        # To align along the outstroke, work out how much the end of the
        # corner pokes out, then find a point on the curve that distance
        # away. Use that as the vector
        distance = self.last_seg[-1].position.y if self.flipped else self.last_seg[-1].position.x
        t = point_on_seg_at_distance(as_tuples(self.outstroke), distance)
        outstroke_intersection_point = segmentPointAtT(as_tuples(self.outstroke), t)
        outstroke_angle = math.atan2(
            outstroke_intersection_point[1] - self.target_node.position.y,
            outstroke_intersection_point[0] - self.target_node.position.x,
        )

        # And the same for the instroke, determined by the Y value of
        # the first point on the corner component
        distance = -self.first_seg[0].position.x if self.flipped else self.first_seg[0].position.y
        t2 = point_on_seg_at_distance(as_tuples(self.instroke), distance)
        instroke_intersection_point = segmentPointAtT(
            as_tuples(reversed(self.instroke)), t2
        )
        instroke_angle = math.atan2(
            self.target_node.position.y - instroke_intersection_point[1],
            self.target_node.position.x - instroke_intersection_point[0],
        )
        instroke_angle += math.radians(90)

        if self.alignment == Alignment.OUTSTROKE:
            angle += outstroke_angle
        elif self.alignment == Alignment.INSTROKE:
            angle += instroke_angle
        elif self.alignment == Alignment.MIDDLE:
            angle += (instroke_angle + outstroke_angle) / 2
        else:  # Unaligned, do nothing
            pass

        # Rotate the paths around the origin and then align them
        # so that the origin of the corner starts on the target node
        rot = Transform().rotate(angle)
        translation = Transform().translate(
            self.target_node.position.x + self.effective_start[0], self.target_node.position.y
        )
        transform = translation.transform(rot)
        # transform = rot.transform(translation)
        for path in [self.corner_path] + self.other_paths:
            for node in path.nodes:
                node.position.x, node.position.y = transform.transformPoint((node.position.x, node.position.y))

        return instroke_intersection_point, outstroke_intersection_point

    def recompute_instroke_intersection_point(self):
        return unbounded_seg_seg_intersection(
            as_tuples(self.first_seg[0:2]), as_tuples(self.instroke)
        )

    def recompute_outstroke_intersection_point(self, original_outstroke):
        if self.flipped:
            # Project it
            return unbounded_seg_seg_intersection(
                as_tuples(self.last_seg), original_outstroke
            )

        # Bend it
        return closest_point_on_segment(
            original_outstroke,
            (self.corner_path.nodes[-1].position.x, self.corner_path.nodes[-1].position.y),
        )

    def split_instroke(self, intersection):
        if len(self.instroke) == 2:
            # Splitting a line is easy...
            (
                self.path.nodes[self.target_node_ix].position.x,
                self.path.nodes[self.target_node_ix].position.y,
            ) = otRound(intersection[0]), otRound(intersection[1])
        else:
            new_cubic = split_cubic_at_point(
                as_tuples(self.instroke), intersection, inward=True
            )
            for new_pt, old in zip(new_cubic, self.instroke):
                old.position.x, old.position.y = otRound(new_pt[0]), otRound(new_pt[1])
        assert self.instroke[-1].position.x, self.instroke[-1].position.x == intersection

    def fixup_outstroke(self, original_outstroke, intersection):
        # Split the outstroke at the nearest point to the intersection.
        # The outstroke has moved now, since we have inserted the path
        outstroke = get_next_segment(
            self.path,
            (self.target_node_ix + len(self.corner_path.nodes) - 1) % len(self.path.nodes),
        )

        if len(outstroke) == 2:
            (outstroke[0].position.x, outstroke[0].position.y) = otRound(intersection[0]), otRound(
                intersection[1]
            )
        else:
            new_cubic = split_cubic_at_point(
                original_outstroke, intersection, inward=False
            )
            for new_pt, old in zip(new_cubic, outstroke):
                old.position.x, old.position.y = otRound(new_pt[0]), otRound(new_pt[1])

    def stretch_first_seg_to_fit(self, intersection):
        delta = (
            intersection[0] - self.first_seg[0].position.x,
            intersection[1] - self.first_seg[0].position.y,
        )
        self.first_seg[1].position.x += delta[0]
        self.first_seg[1].position.y += delta[1]

    def reverse_corner_path(self):
        # XXX This is not how you reverse a GSPath
        self.corner_path.nodes = list(reversed(self.corner_path.nodes))

    def insert_other_paths(self):
        for path in self.other_paths:
            for node in path.nodes:
                otRoundNode(node)
            self.layer.paths.append(path)


def process_font(gsfont, only_glyph=None):
    for glyph in gsfont.glyphs:
        if only_glyph and glyph.name != only_glyph:
            continue
        for layer in glyph.layers:
            if not layer.layerId or not layer.associatedMasterId:
                continue
            layer_hints = [CornerComponentApplier(
                hint, glyph, layer
                ) for hint in layer.hints if hint.type == "Corner"]
            for cc in layer_hints:
                cc.apply()
            # Remove the hints!
            layer.hints = [hint for hint in layer.hints if hint.type != "Corner"]
