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