import subprocess
import os
import trimesh

def make_stl(angle): #get the mesh of crocus with certain angle repective to the satellite
    with open("tmp.scad", "w") as f:
        f.write("""angle = {};
translate([0,0,-5]) union() {{
    translate([-5,-5,0]){{
        cube([10,10,20]);
    }}
    for (i = [1:2]) {{
        rotate([0,0,180*i]) translate([-4.5,0,0]) rotate([0,angle,0]) translate([0,0,-10]){{
                cube([1,10,20], true);
        }}
    }}
}}""".format(angle))
    subprocess.call(['openscad', '-o', str(angle)+'.stl', 'tmp.scad'])
    os.remove('tmp.scad')

for angle in range(0,60,5):
    make_stl(angle)
