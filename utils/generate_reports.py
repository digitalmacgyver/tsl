#!/usr/bin/python

import json
import re
import sys

import tsl.script.Presences
import tsl.script.Interactions
import tsl.script.Script
import tsl.script.Structure

from tsl.script.parse.const import CHARACTER, DISCUSS, LOCATION, SETTING
from tsl.script.reports.reports import top_presences, top_interactions, get_presence_csv, get_interaction_csv


scripts = [
    ( 'The Big Lebowski', '../example-scripts/the_big_lebowski.txt' ),
    ( 'Chinatown', '../example-scripts/chinatown.txt' ),
    ( 'Dune', '../example-scripts/dune.txt' ),
    ( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    ( 'The Matrix', '../example-scripts/the_matrix.txt' ),
    ]

def process_script( script ):
    #import pdb
    #pdb.set_trace()

    name = script[0]
    script_file = script[1]

    print "Working on:", name

    outdir = '../example-scripts/parsed/' + re.sub( r'\s+', '_', name.lower() )

    Script = tsl.script.Script.Script( name, outdir )
    Script.load()

    Structure = tsl.script.Structure.Structure( name, outdir )
    Structure.load()

    Presences = tsl.script.Presences.Presences( name, outdir )
    Presences.load()

    Interactions = tsl.script.Interactions.Interactions( name, outdir )
    Interactions.load()

    presence_png = tsl.script.reports.reports.presence_plot( Script, map( lambda x: x[0], top_presences( Presences, top_n=8, noun_types=[CHARACTER] ) ), "Top 8 Character Presence in "+name )
    f = open( outdir+'/character_presence.png', 'w' )
    f.write( presence_png.getvalue() )
    presence_png.close()
    presence_png = None
    f.close()

    f = open( outdir+'/presence.csv', 'w' )
    f.write( get_presence_csv( Presences ) )
    f.close()

    f = open( outdir+'/interaction.csv', 'w' )
    f.write( get_interaction_csv( Presences, Interactions ) )
    f.close()

    output_top_presences( top_presences( Presences, top_n=5, noun_types=[CHARACTER] ), outdir+'/top5_characters.csv' )
    output_top_presences( top_presences( Presences, top_n=5, presence_types=[DISCUSS] ), outdir+'/top5_speakers.csv' )
    output_top_presences( top_presences( Presences, top_n=5, noun_types=[LOCATION] ), outdir+'/top5_locations.csv' )

    output_top_interactions( top_interactions( Presences, Interactions, top_n=5, interaction_types=[SETTING] ), outdir+'/top5_hangouts.csv' )
    output_top_interactions( top_interactions( Presences, Interactions, top_n=5, noun_types=[ ( CHARACTER, CHARACTER ) ] ), outdir+'/top5_bffs.csv' )
    output_top_interactions( top_interactions( Presences, Interactions, top_n=5, interaction_types=[DISCUSS] ), outdir+'/top5_speakers.csv' )

def output_top_presences( presences, filename ):
    f = open( filename, 'w' )
    f.write("name,noun_type,appearances\n")
    for p in presences:
        f.write( ','.join( [ p[0], p[1], str( p[2] ) ] ) )
        f.write( "\n" )
    f.close()

def output_top_interactions( interactions, filename ):
    f = open( filename, 'w' )
    f.write("name1,name2,interactions\n")
    for i in interactions:
        f.write( ','.join( [ i[0], i[1], str( i[2] ) ] ) )
        f.write( "\n" )
    f.close()

for script in scripts:
    process_script( script )
