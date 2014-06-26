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
    #( 'Vertigo', '../example-scripts/vertigo.txt' ),
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
    #( 'Faults', '../example-scripts/faults.txt' ),
    #( 'Extinction', '../example-scripts/extinction.txt' ),
    #( 'Elsewhere', '../example-scripts/elsewhere.txt' ),
    #( 'Dude', '../example-scripts/dude.txt' ),
    #( 'Dogfight', '../example-scripts/dogfight.txt' ),
    #( 'Diablo Run', '../example-scripts/diablo_run.txt' ),
    #( 'Clarity', '../example-scripts/clarity.txt' ),
    #( 'Cake', '../example-scripts/cake.txt' ),
    #( 'A Beautiful Day in the Neighborhood', '../example-scripts/a_beautiful_day_in_the_neighborhood.txt' ),
    #( 'American Sniper', '../example-scripts/american_sniper.txt' ),
    #( 'A Monster Calls', '../example-scripts/a_monster_calls.txt' ),
    #( 'Beast', '../example-scripts/beast.txt' ),
    #( 'Beauty Queen', '../example-scripts/beauty_queen.txt' ),
    #( 'Broken Cove', '../example-scripts/broken_cove.txt' ),
    #( 'Burn Site', '../example-scripts/burn_site.txt' ),
    #( 'Bury the Lead', '../example-scripts/bury_the_lead.txt' ),
    #( 'Pox Americana', '../example-scripts/pox_americana.txt' ),
    #( 'Prisoners', '../example-scripts/prisoners.txt' ),
    #( 'Pure', '../example-scripts/pure.txt' ),
    #( 'Queen of Hearts', '../example-scripts/queen_of_hearts.txt' ),
    #( 'Randle is Benign', '../example-scripts/randle_is_benign.txt' ),
    #( 'Real Steel', '../example-scripts/real_steel.txt' ),
    #( 'Reminiscence', '../example-scripts/reminiscence.txt' ),
    #( 'Revelations', '../example-scripts/revelations.txt' ),
    #( 'Rush', '../example-scripts/rush.txt' ),
    #( 'Rust and Bone', '../example-scripts/rust_and_bone.txt' ),
    #( 'Skyfall', '../example-scripts/skyfall.txt' ),
    #( 'Smashed', '../example-scripts/smashed.txt' ),
    #( 'Snow White and the Huntsman', '../example-scripts/snow_white_and_the_huntsman.txt' ),
    #( 'The Croods', '../example-scripts/the_croods.txt' ),
    #( 'Fixer', '../example-scripts/fixer.txt' ),
    #( 'Free Byrd', '../example-scripts/free_byrd.txt' ),
    ( 'Frisco', '../example-scripts/frisco.txt' ),
    #( 'From Here to Albion', '../example-scripts/from_here_to_albion.txt' ),
    #( 'Fully Wrecked', '../example-scripts/fully_wrecked.txt' ),
    #( 'Holland Michigan', '../example-scripts/holland_michigan.txt' ),
    #( 'Im So Proud of You', '../example-scripts/im_so_proud_of_you.txt' ),
    #( 'Ink and Bone', '../example-scripts/ink_and_bone.txt' ),
    #( 'Inquest', '../example-scripts/inquest.txt' ),
    #( 'Ipoy Master', '../example-scripts/ipoy_master.txt' ),
    #( 'Last Minute Maids', '../example-scripts/last_minute_maids.txt' ),
    #( 'Line of Duty', '../example-scripts/line_of_duty.txt' ),
    #( 'Make a Wish', '../example-scripts/make_a_wish.txt' ),
    #( 'Man of Sorrow', '../example-scripts/man_of_sorrow.txt' ),
    #( 'Nicholas', '../example-scripts/nicholas.txt' ),
    #( 'Patient Z', '../example-scripts/patient_z.txt' ),
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
    
