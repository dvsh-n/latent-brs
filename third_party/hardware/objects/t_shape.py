# pip install solidpython
# Install OpenSCAD separately, then run:
#   python make_t_shell_and_lid.py
# It will write .scad; open in OpenSCAD and export STL (or use openscad CLI).

from solid import *
from solid.utils import *
import os

# ---------- Parameters (mm) ----------
# Overall T dimensions in XY
bar_len  = 120.0   # along X
bar_w    = 30.0   # along Y
stem_len = 90.0 + bar_w   # along +Y
stem_w   = 30.0   # along X

T_scale = 5/3
stem_len = int(stem_len * T_scale)
stem_w = int(stem_w   * T_scale)
bar_len = int(bar_len  * T_scale)
bar_w = int(bar_w    * T_scale)

# Heights
H_total  = 40.0   # total outer height (Z)

# Shell
wall     = 4    # side wall thickness
floor    = 4    # bottom thickness
clearance = 0.50  # lid lip clearance

# Lid
lid_th   = 4.0    # top plate thickness
lip_h    = 8.0    # how deep the lip goes into cavity
lip_t    = 4.0    # lip wall thickness (before clearance)

# ---------- Helpers ----------
def t_2d(stem_len, stem_w, bar_len, bar_w):
    """
    Returns a 2D T shape centered at origin.
    Stem extends in +Y, bar is centered near the top.
    """
    stem = translate([0, stem_len/2])(square([stem_w, stem_len], center=True))
    bar  = translate([0, stem_len - bar_w/2])(square([bar_len, bar_w], center=True))
    return union()(stem, bar)

def linear_extrude_z(shape2d, h):
    return linear_extrude(height=h)(shape2d)

# ---------- Build outer body ----------
outer2d = t_2d(stem_len, stem_w, bar_len, bar_w)
outer3d = linear_extrude_z(outer2d, H_total)

# ---------- Build inner cavity (subtract) ----------
# Inner dims: shrink by 2*wall in X and Y directions
inner_stem_len = stem_len - 2*wall
inner_stem_w   = stem_w   - 2*wall
inner_bar_len  = bar_len  - 2*wall
inner_bar_w    = bar_w    - 2*wall

# Guard against invalid params
if min(inner_stem_len, inner_stem_w, inner_bar_len, inner_bar_w) <= 0:
    raise ValueError("Wall too thick for given T dimensions.")

inner2d = t_2d(inner_stem_len, inner_stem_w, inner_bar_len, inner_bar_w)
y_shift = (stem_len - wall) - inner_stem_len   # align inner top to (outer top - wall)
inner2d = translate([0, y_shift])(inner2d)
# Shift cavity up so we keep a floor thickness
inner3d = translate([0, 0, floor])(linear_extrude_z(inner2d, H_total - floor + 1e-3))

shell = difference()(outer3d, inner3d)

# ---------- Lid: plate + lip ----------
# Lid plate matches outer footprint
plate = translate([0, 0, H_total])(linear_extrude_z(outer2d, lid_th))

# Lip fits inside cavity:
# Use slightly smaller footprint than inner cavity (by clearance)
lip_stem_len = inner_stem_len - 2*clearance
lip_stem_w   = inner_stem_w   - 2*clearance
lip_bar_len  = inner_bar_len  - 2*clearance
lip_bar_w    = inner_bar_w    - 2*clearance

lip_outer2d = t_2d(lip_stem_len, lip_stem_w, lip_bar_len, lip_bar_w)

# Make the lip itself hollow so it isn't a solid block:
lip_inner_stem_len = lip_stem_len - 2*lip_t
lip_inner_stem_w   = lip_stem_w   - 2*lip_t
lip_inner_bar_len  = lip_bar_len  - 2*lip_t
lip_inner_bar_w    = lip_bar_w    - 2*lip_t

if min(lip_inner_stem_len, lip_inner_stem_w, lip_inner_bar_len, lip_inner_bar_w) <= 0:
    raise ValueError("lip_t too thick; reduce lip_t or increase T dimensions.")

lip_inner2d = t_2d(lip_inner_stem_len, lip_inner_stem_w, lip_inner_bar_len, lip_inner_bar_w)
y_shift = (lip_stem_len - wall) - lip_inner_stem_len   # align inner top to (outer top - wall)
lip_inner2d = translate([0, y_shift])(lip_inner2d)

lip_ring2d = difference()(lip_outer2d, lip_inner2d)
lip = translate([0, 0, H_total - lip_h])(linear_extrude_z(lip_ring2d, lip_h))

lip = translate([0, y_shift, 0])(lip)  # align lip Y with cavity
lid = union()(plate, lip)

# ---------- Output ----------
# Separate parts side-by-side for easy export
assembly = union()(
    translate([-120, 0, 0])(shell),
    translate([+120, 0, 0])(lid),
)

out_dir = "hardware/objects/"
os.makedirs(out_dir, exist_ok=True)


scad_render_to_file(shell, os.path.join(out_dir, "t_shell.scad"), file_header="$fn=64;")
scad_render_to_file(lid,   os.path.join(out_dir, "t_lid.scad"),   file_header="$fn=64;")
scad_render_to_file(assembly, os.path.join(out_dir, "t_shell_and_lid_preview.scad"), file_header="$fn=64;")

print(f"Wrote .scad files to {out_dir}")
print("Open in OpenSCAD and Export STL for each part.")
