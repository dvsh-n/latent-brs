from solid import *
from solid.utils import *
import math
import os

# ---------- Parameters (mm) ----------
# Pusher rod
rod_d = 10.0
rod_h = 150.0

# Mounting flange
flange_d = 70.0         # outer diameter of flange (choose >= BCD + margin)
flange_t = 5.0         # thickness

# Bolt circle / holes
bcd = 50.0              # bolt circle diameter
bolt_r = bcd / 2.0
n_bolts = 2             # "opposite bolts" -> 2; change to 4/6 if needed

# M6 clearance hole (typical)
m6_clear = 6.5          # 6.4~6.6 mm is common for FDM prints
hole_d = m6_clear

# Optional: add a center hole (for cable, alignment, or tool)
center_hole_d = 0.0     # set to e.g. 10.0 if you want
# Fillets/chamfers (approx via Minkowski is heavy; keep simple)
edge_chamfer = 0.0

# Add counterbores for socket head screws (optional but recommended)
head_d = 13.0       # ~10 mm head + clearance
head_h = 3.0        # recess depth

# ---------- Helpers ----------
def bolt_holes(n, radius, d, h):
    holes = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        x = radius * math.cos(ang)
        y = radius * math.sin(ang)
        holes.append(translate([x, y, 0])(cylinder(d=d, h=h)))
    return union()(*holes)

# ---------- Build parts ----------
# Flange base (centered at origin, z from 0..flange_t)
flange = cylinder(d=flange_d, h=flange_t)

# Rod: place on top of flange, centered
rod = translate([0, 0, flange_t])(cylinder(d=rod_d, h=rod_h))

body = union()(flange, rod)

# # Subtract bolt holes through flange
# holes = bolt_holes(n_bolts, bolt_r, hole_d, flange_t + 1.0)

# Subtract bolt clearance holes through flange
holes_thru = bolt_holes(n_bolts, bolt_r, hole_d, flange_t + 1.0)


counterbore = bolt_holes(n_bolts, bolt_r, head_d, head_h)

# Place counterbore from the top face of flange downward
counterbore = translate([0, 0, flange_t - head_h])(counterbore)

holes = union()(holes_thru, counterbore)

# Optional center hole
if center_hole_d and center_hole_d > 0:
    holes = union()(holes, cylinder(d=center_hole_d, h=flange_t + 1.0))


final = difference()(body, holes)

# ---------- Output ----------
out_dir = "hardware/objects/"
os.makedirs(out_dir, exist_ok=True)

scad_render_to_file(final, os.path.join(out_dir, "pusher.scad"), file_header="$fn=96;")
print(f"Wrote {os.path.join(out_dir, 'pusher.scad')} (render in OpenSCAD, then export STL).")