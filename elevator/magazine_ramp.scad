use <MCAD/regular_shapes.scad>
include <magazine.scad>

//translate([PIECE_D/2,0,0])
//%piece();

module ramp() {

    RAMP_THICKNESS = 3;
    RAMP_RISE = 30;
    RAMP_RUN = PIECE_D + 20;

    // Ramp
    translate([RAMP_THICKNESS, 0, 0])
    hull() {
        cube([PIECE_H, 1, RAMP_THICKNESS]);

        translate([0, RAMP_RUN, RAMP_RISE - THICKNESS])
        cube([PIECE_H, 1, RAMP_THICKNESS]);
    }

    // Walls
    TOP_HEIGHT = 30;
    // left wall
    hull() {
        cube([RAMP_THICKNESS, RAMP_RUN + 1, RAMP_RISE]);

        translate([0, 0, RAMP_RISE + TOP_HEIGHT])
        #cube([0.1, RAMP_RUN, 0.1]);
    }

    // back wall
    translate([0, RAMP_RUN - RAMP_THICKNESS + 1])
    cube([PIECE_H + RAMP_THICKNESS, RAMP_THICKNESS, RAMP_RISE + TOP_HEIGHT]);

    // right wall
    translate([RAMP_THICKNESS + PIECE_H, 0, 0])
    cube([RAMP_THICKNESS, RAMP_RUN + 1, RAMP_RISE + TOP_HEIGHT]);


    // Magazine sleeve
    SLEEVE_WIDTH = 12;
    SLEEVE_HEIGHT = RAMP_RISE;
    translate([-PIECE_D/2 - THICKNESS, RAMP_RUN/2, 0])
    {
        difference(){
            cylinder_tube(SLEEVE_HEIGHT, PIECE_D/2 + 2 * THICKNESS, THICKNESS);

            SLEEVE_CUT = 1000;
            translate([-SLEEVE_CUT/2 + SLEEVE_WIDTH, 0, 0])
            cube([SLEEVE_CUT, SLEEVE_CUT, SLEEVE_CUT], center=true);
        }

        %mag();
    }

}

module rotramp(height, dz, angle, width, depth) {
//    translate([width/2, depth/2, dz/2])
    for (z = [0:dz:height-dz]) {
        hull() {
            for (i = [0,1]) {
                rotate([0,0,(z + i*dz)/height * -angle])
                translate([0, 0, z + i*dz])
                cube([width, depth, dz], center=false);
            }
        }
    }
}

//RAMP_RUN = PIECE_D * 1.5;
//RAMP_ANGLE = 90;
//RAMP_WIDTH = PIECE_D;

//cube([THICKNESS, RAMP_RUN, PIECE_D]);

RAMP_RUN = PIECE_D;
RAMP_HEIGHT_GUESS = PIECE_D + 12; // This depends on STRETCH
STRETCH = 15;
RAMP_WIDTH = PIECE_D * 0.75;


SLIDEY_COEFF = 1.9;
RAMP_ANGLE=30;
intersection() {
    scale([1, 1, 1.2])
    translate([6.4, 0])
    translate([0, 0, 1.2 * RAMP_WIDTH])
    rotate([0, 90 - RAMP_ANGLE])
    translate([SLIDEY_COEFF * RAMP_WIDTH - 10, 0, 0])
    mirror([1,0,0])
    rotate([-90, 0, 0])
    rotramp(RAMP_RUN, $preview ? 1 : 0.05, RAMP_ANGLE, SLIDEY_COEFF * RAMP_WIDTH, THICKNESS);

    cube(100);
}

//translate([30, 0, 0])
//mirror([1,0])
//rotate([0, 30, 10])
//cube([THICKNESS, RAMP_RUN, RAMP_HEIGHT_GUESS]);

translate([RAMP_WIDTH, 0])
rotate([0, 90, 0])
%piece();

//translate([0, RAMP_RUN, 0])
//rotate([90, 0, 0])
//intersection()  {
//    translate([-STRETCH, 0, 0])
//    oval_prism(RAMP_RUN, PIECE_D + STRETCH - THICKNESS/2, RAMP_WIDTH + STRETCH);
//    cube(10 * STRETCH);
//}

//mirror([1,0])
//cube([THICKNESS, RAMP_RUN, RAMP_HEIGHT_GUESS + PIECE_H]);

CUPPER_CUTOFF = 15;
CUPPER_LOW_HEIGHT = 30;

module cupper_bounds() {
    translate([0, -CUPPER_CUTOFF])
    hull() {
        cube([1, CUPPER_CUTOFF, RAMP_HEIGHT_GUESS]);
        translate([PIECE_D,0])
        cube([1, CUPPER_CUTOFF, CUPPER_LOW_HEIGHT]);
    }
}

intersection() {
    translate([PIECE_D/2, -PIECE_D/2 - 2*THICKNESS]) {
//        %mag();
        cylinder_tube(RAMP_HEIGHT_GUESS, (PIECE_D + 4 * THICKNESS)/2, THICKNESS);
    }

    cupper_bounds();
}

intersection() {
    mirror([0, 1, 0])
    cube([PIECE_D, THICKNESS, RAMP_HEIGHT_GUESS]);
    cupper_bounds();
}

intersection() {
    cupper_bounds();
    mirror([0,1,0])
    cube([THICKNESS, CUPPER_CUTOFF - 6, RAMP_HEIGHT_GUESS]);
}

intersection() {
    cupper_bounds();
    translate([THICKNESS, 0])
    mirror([0,1,0])
    cube([THICKNESS, CUPPER_CUTOFF - 8.5, RAMP_HEIGHT_GUESS]);
}

intersection() {
    cupper_bounds();
    translate([PIECE_D - THICKNESS, 0])
    mirror([0,1,0])
    cube([THICKNESS, CUPPER_CUTOFF - 6, RAMP_HEIGHT_GUESS]);
}

intersection() {
    cupper_bounds();
    translate([PIECE_D - 2*THICKNESS, 0])
    mirror([0,1,0])
    cube([THICKNESS, CUPPER_CUTOFF - 8.5, RAMP_HEIGHT_GUESS]);
}

cube([20, RAMP_RUN, THICKNESS]);

hull(){
translate([0, RAMP_RUN, 0])
cube([25, THICKNESS, PIECE_D]);

cube([THICKNESS, RAMP_RUN, PIECE_D]);
}
translate([RAMP_WIDTH, 0, 0]) {
    mirror([0,1])
    cube([PIECE_H, THICKNESS, CUPPER_LOW_HEIGHT]);

    hull() {
        mirror([0,1])
        translate([-10, 0, 10])
        cube([PIECE_H+10, 1, THICKNESS]);

        translate([-10, RAMP_RUN, 0])
        cube([PIECE_H+10, 1, THICKNESS]);

        cube([PIECE_H, RAMP_RUN, 1]);
    }

    translate([PIECE_H, -10, 0])
    cube([THICKNESS, RAMP_RUN + 10, 40]);

    // Grabbing platform
    translate([0, RAMP_RUN, 0])
    {
        GRAB_LEN = PIECE_D / 1.333;
        cube([PIECE_H, GRAB_LEN, THICKNESS]);

        translate([-THICKNESS, 0, 0])
        cube([THICKNESS, GRAB_LEN, PIECE_D / 2]);

        translate([PIECE_H, 0, 0])
        cube([THICKNESS, GRAB_LEN, PIECE_D / 2]);

        translate([-THICKNESS, GRAB_LEN, 0])
        cube([PIECE_H + 2*THICKNESS, THICKNESS, PIECE_D / 3]);
    }
}

//    cube([1, 1, 1])

//translate([PIECE_D/2,0]) piece();
/////////////////
// OTHER STUFF //
/////////////////

// Width of plastic along top of board
RIM_WIDTH = 3.5;
// Dimensions of holes for game pieces
DROP_WIDTH = 8.85;
DROP_LENGTH = 33.57;
// Dimensions of board addon slot
SLOT_WIDTH = 4.3;
SLOT_LENGTH = 11;
//SLOT_RADIUS = 1;
//SLOT_DIAM = SLOT_RADIUS * 2;

module slot_leg_test() {
    cube([SLOT_WIDTH, SLOT_LENGTH, 20]);
}

module board_leg() {
    LEG_HEIGHT = 150;
    LEG_WIDTH = 17;
    LEG_DEPTH = 6;

    THICK_HEIGHT = LEG_HEIGHT - 50;
    THICK_WIDTH = LEG_WIDTH;
    THICK_DEPTH = 12;

    LEG_SLOT_MALE_HEIGHT = 5;
    CHANNEL_WIDTH = 4;
    CHANNEL_DEPTH = 8;

    CHANNEL_START_HEIGHT = LEG_HEIGHT - 20;
    CHANNEL_GRAB_HEIGHT = LEG_HEIGHT - 85;
    CHANNEL_GRAB_THICKNESS = 1.5;

    difference() {
        // main body
        hull() {
            cube([LEG_WIDTH, LEG_DEPTH, LEG_HEIGHT]);
            cube([THICK_WIDTH, THICK_DEPTH, THICK_HEIGHT]);
        }

        // channel
        translate([(LEG_WIDTH - CHANNEL_WIDTH)/2, -1, -1])
        cube([CHANNEL_WIDTH, CHANNEL_DEPTH + 1, CHANNEL_START_HEIGHT + 1]);
    }

    // channel grab
    translate([(LEG_WIDTH - CHANNEL_WIDTH)/2 - 1,0, CHANNEL_GRAB_HEIGHT - 10])
    cube([CHANNEL_WIDTH + 2, CHANNEL_GRAB_THICKNESS, 10]);

    // slot male end
    translate([LEG_WIDTH/2 - SLOT_LENGTH/2,0,LEG_HEIGHT])
    cube([SLOT_LENGTH, SLOT_WIDTH, LEG_SLOT_MALE_HEIGHT]);
}

//board_leg();