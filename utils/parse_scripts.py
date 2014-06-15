#!/usr/bin/python

import json
import re
import sys

import tsl.script.parse.load
import tsl.script.parse.parse
from tsl.script.parse.const import STRICT
import tsl.script.Presences
import tsl.script.Interactions
import tsl.script.Script
import tsl.script.Structure

scripts = [
    #( 'Chinatown', '../example-scripts/chinatown.txt' ),
    # ( 'Dune', '../example-scripts/dune.txt' ),
    # ( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    # ( 'The Matrix', '../example-scripts/the_matrix.txt' ),
    # ( 'Good Will Hunting', '../example-scripts/good_will_hunting.txt' ),
    # ( 'The Book of Eli', '../example-scripts/the_book_of_eli.txt' ),
    # ( 'Starwars', '../example-scripts/starwars.txt' ),
    # ( 'Alien', '../example-scripts/alien.txt' ),
    # ( 'Vertigo', '../example-scripts/vertigo.txt' ),
    # ( 'Terminator 2', '../example-scripts/terminator_2.txt' ),
    # ( 'Ratatouille', '../example-scripts/ratatouille.txt' ),
     # Questionable formatting
    # ( 'Analyze That', '../example-scripts/analyze_that.txt' ),
    # ( 'Batman Begins', '../example-scripts/batman_begins.txt' ),
    # ( 'Death to Smoochy', '../example-scripts/death_to_smoochy.txt' ),
    # ( 'Get Carter', '../example-scripts/get_carter.txt' ),
    # ( 'Gothika', '../example-scripts/gothika.txt' ),
    # ( 'Groundhogs Day', '../example-scripts/groundhogs_day.txt' ),
    # ( 'Red Planet', '../example-scripts/red_planet.txt' ),
    # ( 'Smurfs', '../example-scripts/smurfs.txt' ),
    # ( 'Sweet November', '../example-scripts/sweet_november.txt' ),
    # ( 'Taking Lives', '../example-scripts/taking_lives.txt' ),
    # ( 'Thirteen Ghosts', '../example-scripts/thirteen_ghosts.txt' ),
    # ( '42', '../example-scripts/42.txt' ),
    #( 'Frozen', '../example-scripts/frozen.txt' ),
    #( 'Fruitvale Station', '../example-scripts/fruitvale_station.txt' ),
    #( 'Recursion', '../example-scripts/recursion-05-16-14.txt' ),
    #( 'Recursion', '../example-scripts/recursion-06-10-14.txt' ),
    #( 'Starthur', '../example-scripts/starthur.txt' ),
    #( 'All is Lost', '../example-scripts/all_is_lost.txt' ),
    #( 'Amour', '../example-scripts/amour.txt' ),
    #( 'Argo', '../example-scripts/argo.txt' ),
    #( 'August Osage County', '../example-scripts/august_osage_county.txt' ),
    #( 'Celest and Jesse Forever', '../example-scripts/celeste_and_jesse_forever.txt' ),
    #( 'Chronicle', '../example-scripts/chronicle.txt' ),
    #( 'Dallas Buyers Club', '../example-scripts/dallas_buyers_club.txt' ),
    #( 'Despicable Me 2', '../example-scripts/despicable_me_2.txt' ),
    #( 'The Wolf of Wall Street', '../example-scripts/the_wolf_of_wall_street.txt' ),
    #( 'Prince of Persia', '../example-scripts/prince_of_persia.txt' ),
    #( 'Oz the Great and Powerful', '../example-scripts/oz_the_great_and_powerful.txt' ),
    #( 'Nebraska', '../example-scripts/nebraska.txt' ),
    #( 'Monsters University', '../example-scripts/monsters_university.txt' ),
    #( 'Magic Mike', '../example-scripts/magic_mike.txt' ),
    #( 'Lone Survivor', '../example-scripts/lone_survivor.txt' ),
    #( 'Kill Your Darlings', '../example-scripts/kill_your_darlings.txt' ),
    #( 'Kick Ass 2', '../example-scripts/kick_ass_2.txt' ),
    #( '1969 A Space Odyssy', '../example-scripts/1969_a_space_odyssey.txt' ),
    #( 'The Great Gatsby', '../example-scripts/the_great_gatsby.txt' ),
    #( 'The Invisible Woman', '../example-scripts/the_invisible_woman.txt' ),
    #( 'The Past', '../example-scripts/the_past.txt' ),
    #( 'Twilight', '../example-scripts/twilight.txt' ),
    #( 'Wadjda', '../example-scripts/wadjda.txt' ),
    #( 'Woman in Black', '../example-scripts/woman_in_black.txt' ),
    ]


def process_script( script, parse_mode=STRICT ):

    name = script[0]
    script_file = script[1]

    print "Working on:", name

    outdir = '../example-scripts/parsed/' + re.sub( r'\s+', '_', name.lower() )

    f = open( script_file, 'r' )
    body = f.readlines()
    body = [ unicode( x, errors='ignore' ) for x in body ]
    script_lines = tsl.script.parse.load.load_txt( body, lines_per_page = 56 )
    f.close()

    s = tsl.script.Script.Script( name, outdir )
    s.script_lines = script_lines
    s.save()

    script_structure = tsl.script.parse.parse.parse_script_lines( s )

    script_structure.save()

    ( Presences, Interactions ) = tsl.script.parse.parse.compute_presence_and_interactions( s, script_structure, parse_mode=parse_mode )

    Presences.save()
    Interactions.save()

for script in scripts:
    process_script( script, parse_mode=STRICT )
    
