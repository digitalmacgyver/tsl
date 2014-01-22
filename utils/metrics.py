#!/usr/bin/python

import json
import nltk
import numpy
import re
import sys

import tsl.script.Presences
import tsl.script.Interactions
import tsl.script.Script
import tsl.script.Structure

from tsl.script.parse.const import CHARACTER, DISCUSS, LOCATION, SETTING
from tsl.script.reports.reports import top_presences, top_interactions, get_presence_csv, get_interaction_csv


scripts = [
    #( 'The Big Lebowski', '../example-scripts/the_big_lebowski.txt' ),
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

    f = open( script_file )
    raw_script = f.read()

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

    output = {}

    top_characters = top_presences( Presences, noun_types=[CHARACTER] )
    top_locations = top_presences( Presences, noun_types=[LOCATION] )

    # Distinct words.
    all_words = nltk.wordpunct_tokenize( raw_script )
    word_tokens = [ t for t in all_words if re.search( r'\w', t ) ]
    text = nltk.Text( word_tokens )
    words = [ w.lower() for w in text ]
    vocab = sorted( set( words ) )
    output['distinct_words'] = len( vocab )
    
    #import pdb
    #pdb.set_trace()

    # Characters who speak to main character
    speakers = top_interactions( Presences, Interactions, noun_types=[( CHARACTER, CHARACTER )], interaction_types=[DISCUSS], names=[top_characters[0][0]] )
    output['main_character_interlocutor_count'] = len( speakers )
    
    # Distinct locations
    output['distinct_locations'] = len( top_locations )

    
    # Percentage of scenes with main character
    scenes = Presences.presence_sn.keys()
    scene_count = len( scenes )
    main_character_appearances = 0
    for scene in scenes:
        if top_characters[0][0] in Presences.presence_sn[scene]:
            main_character_appearances += 1
    output['precentage_of_scenes_with_main_character'] = float( main_character_appearances ) / scene_count
    output['main_character'] = top_characters[0][0]

    # Characters speaking in scene.
    scene_talkers = {}
    for presence in Presences.presences:
        if presence['presence_type'] == DISCUSS:
            scene = presence['where']['scene_id']
            character = presence['name']
            if scene in scene_talkers:
                scene_talkers[scene][character] = True
            else:
                scene_talkers[scene] = { character : True }
    scene_talker_data = []
    for scene in scene_talkers.keys():
        scene_talker_data.append( ( scene, len( scene_talkers[scene].keys() ) ) )
        
    for scene_id in Presences.presence_sn.keys():
        if scene_id not in scene_talkers:
            scene_talker_data.append( ( scene_id, 0 ) )

    scene_talker_data = sorted( scene_talker_data, key=lambda a: int( a[0] ) )
    
    output['characters_speaking_in_scene_stats'] = get_stats( scene_talker_data )
    #output['characters_speaking_in_scene'] = scene_talker_data

    print json.dumps( output, sort_keys=True, indent=4 )


def get_stats( data ):
    '''Input is an unsorted array of ( 'scene_id', numerical quantity
    ) tuples.  We compute the average, min, max, median, and standard
    deviation of the numerical quantities.'''
    
    numbers = [ x[1] for x in data ]
    return { 'average' : numpy.average( numbers ),
             'min'     : min( numbers ),
             'max'     : max( numbers ),
             'median'  : numpy.median( numbers ),
             'stdev'   : numpy.std( numbers ) }





    # top_presences( Presences, top_n=8, noun_types=[CHARACTER] )
    #output_top_presences( top_presences( Presences, top_n=5, noun_types=[CHARACTER] ), outdir+'/top5_characters.csv' )
    #output_top_presences( top_presences( Presences, top_n=5, presence_types=[DISCUSS] ), outdir+'/top5_speakers.csv' )
    #output_top_presences( top_presences( Presences, top_n=5, noun_types=[LOCATION] ), outdir+'/top5_locations.csv' )
    #output_top_interactions( top_interactions( Presences, Interactions, top_n=5, interaction_types=[SETTING] ), outdir+'/top5_hangouts.csv' )
    #output_top_interactions( top_interactions( Presences, Interactions, top_n=5, noun_types=[ ( CHARACTER, CHARACTER ) ] ), outdir+'/top5_bffs.csv' )
    #output_top_interactions( top_interactions( Presences, Interactions, top_n=5, interaction_types=[DISCUSS] ), outdir+'/top5_speakers.csv' )

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
