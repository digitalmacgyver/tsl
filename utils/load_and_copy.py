#!/usr/bin/python

import json
import re
import sys

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

def process_script( script ):
    name = script[0]
    script_file = script[1]

    print "Working on:", name

    outdir = '../example-scripts/parsed/' + re.sub( r'\s+', '_', name.lower() )

    Script = tsl.script.Script.Script( name, outdir )
    Script.load()
    Script.save( outdir=outdir+'/2' )

    Structure = tsl.script.Structure.Structure( name, outdir )
    Structure.load()
    Structure.save( outdir=outdir+'/2' )

    Presences = tsl.script.Presences.Presences( name, outdir )
    Presences.load()
    Presences.save( outdir=outdir+'/2' )

    Interactions = tsl.script.Interactions.Interactions( name, outdir )
    Interactions.load()
    Interactions.save( outdir=outdir+'/2' )

for script in scripts:
    process_script( script )
    
