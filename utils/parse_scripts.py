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
    ( 'The Big Lebowski', '../example-scripts/the_big_lebowski.txt' ),
    ( 'Chinatown', '../example-scripts/chinatown.txt' ),
    ( 'Dune', '../example-scripts/dune.txt' ),
    ( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    ( 'The Matrix', '../example-scripts/the_matrix.txt' ),
    ]

def process_script( script, parse_mode=STRICT ):

    name = script[0]
    script_file = script[1]

    print "Working on:", name

    outdir = '../example-scripts/parsed/' + re.sub( r'\s+', '_', name.lower() )

    f = open( script_file, 'r' )
    body = f.readlines()
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
    
