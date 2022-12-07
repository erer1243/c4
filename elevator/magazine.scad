//use <MCAD/regular_shapes.scad>

$fn = $preview ? 50 : 300;

WIGGLE_PERCENT = 10;
WIGGLE_COEFF = 1 + WIGGLE_PERCENT/100;
PIECE_D = 32 * WIGGLE_COEFF;
PIECE_H = 7.5 * WIGGLE_COEFF;

module piece() {
    difference() {
        cylinder(h=PIECE_H/WIGGLE_COEFF,
                 d=PIECE_D/WIGGLE_COEFF);

        translate([0, 0, -0.5])
        cylinder(h=1 + PIECE_H/WIGGLE_COEFF,
                 d=PIECE_D/WIGGLE_COEFF * 0.75);
    }
}

module piece_wiggle() { cylinder(h=PIECE_H, d=PIECE_D); }

BOARD_H = 197;
BOARD_W = 259;
BOARD_D = 16.5;

module board() { cube([BOARD_W, BOARD_D, BOARD_H]); }

HOPPER_W = 45;
HOPPER_L = 20;
HOPPER_H = 60;

module hole_negative() {
    scale([1.01, 1.01, 1.01])
    rotate([180, 0, 0])
    translate([0, 0, -HOPPER_H/2])
    hull()
    for (x = [-1, 1]) {
        for (y = [-1, 1]) {
            translate([1.1 * x * PIECE_D/2,
                       1.2 * y * PIECE_H/2,
                       0])
            square_pyramid(5.5, 7, HOPPER_H);
        }
    }
}

module hopper() {
    translate([0, 0, HOPPER_H / 2])
    difference() {
        cube([HOPPER_W, HOPPER_L, HOPPER_H], center=true);

        #
        translate([0, 0, 25])
        cube([HOPPER_W + 5, HOPPER_L + 2, 10], center=true);

        #
        hole_negative();
    }
}

//difference() {
//    hopper();
//    translate([-50, -50, 20])
//    #
//    cube([100, 100, 100]);
//}


ANGLE=40;
INNER_RAD=30;
THICKNESS=3;
PICKLES=30;

module hotdog() {
    rotate_extrude(angle=ANGLE)
    translate([INNER_RAD, 0, 0])
    square([THICKNESS, PIECE_D]);
}

module hamburg() {
    rotate_extrude(angle=ANGLE)
    translate([INNER_RAD, 0, 0])
    square([THICKNESS, PIECE_H]);
}

module pickle(ang, tx, ty) {
    translate([tx, ty, 0])
    rotate([0,0,ang])
    translate([-(INNER_RAD + THICKNESS), 1/2, 0])
    rotate([90, 0, 0])
    rotate_extrude(angle=ANGLE)
    translate([INNER_RAD, 0, 0])
    square(THICKNESS);
}

module hot_hamburg() {
    module pickle_slice() {
        dang = 90/PICKLES;
        for (i = [0:PICKLES-1]) {
            hull() {
                pickle(i*dang, 0, 0);
                pickle((i+1)*dang, 0, 0);
            }
        }
    }

    pickle_slice();

    translate([0, PIECE_H, 0])
    mirror([0, 1, 0])
    pickle_slice();

    translate([0, -(INNER_RAD + THICKNESS), 0])
    rotate([90, 0, 90])
    hotdog();

    translate([-(INNER_RAD + THICKNESS), PIECE_H, 0])
    rotate([90, 0, 0])
    hamburg();
}

module hot_hamburg_sandwich() {
    hot_hamburg();
    translate([PIECE_D, PIECE_H, 0])
    rotate([0, 0, 180])
    hot_hamburg();
}

//scale([1.05, 1.05, 1.05])
//hot_hamburg_sandwich();
//
//translate([PIECE_D/2,PIECE_H,0])
//rotate([90,0,0])
//#piece();

module mag() {
    CAPACITY = 23;
    ANGLE = 160;
    WALLS_HEIGHT = CAPACITY * PIECE_H + THICKNESS;

    translate([0, 0, THICKNESS])
    for (za = [0,180]) {

        rotate([0,0,za])
        rotate_extrude(angle=ANGLE)
        translate([PIECE_D/2, 0, 0])
        square([THICKNESS, WALLS_HEIGHT]);

    }
    cylinder(h=THICKNESS, d=PIECE_D + 2*THICKNESS);

    translate([0, 0, WALLS_HEIGHT])
    rotate_extrude(angle=360)
    translate([PIECE_D/2, 0,0])
    square([THICKNESS, THICKNESS]);
}

//mag();

module string_rider() {
    difference() {
        cylinder(d=PIECE_D / WIGGLE_COEFF, h=PIECE_H);

        CHANNEL_H = PIECE_H / 3;
        #
        translate([0,0,CHANNEL_H/2 + 1])
        cube([PIECE_D * 2, 5, CHANNEL_H], center=true);
    }
}

//translate([0,0,5])
//string_rider();

//*
//for (i = [0:CAPACITY-1]) {
//    translate([0,0,i*PIECE_H])
//    translate([0,0,THICKNESS])
//    #piece();
//}

//piece();
