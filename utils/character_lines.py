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

from tsl.utils.partition import get_dramatic_unit_partitions

scripts = [
#    ( 'Chinatown', '../example-scripts/chinatown.txt' ),
#    ( 'Dune', '../example-scripts/dune.txt' ),
    ( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    ( 'The Matrix', '../example-scripts/the_matrix.txt' ),
#    ( 'Good Will Hunting', '../example-scripts/good_will_hunting.txt' ),
#    ( 'The Book of Eli', '../example-scripts/the_book_of_eli.txt' ),
    ( 'Starwars', '../example-scripts/starwars.txt' ),
#    ( 'Alien', '../example-scripts/alien.txt' ),
#    ( 'Vertigo', '../example-scripts/vertigo.txt' ),
#    ( 'Terminator 2', '../example-scripts/terminator_2.txt' ),
#    ( 'Ratatouille', '../example-scripts/ratatouille.txt' ),
    # Questionable formatting
#    ( 'Analyze That', '../example-scripts/analyze_that.txt' ),
#    ( 'Batman Begins', '../example-scripts/batman_begins.txt' ),
#    ( 'Death to Smoochy', '../example-scripts/death_to_smoochy.txt' ),
#    ( 'Get Carter', '../example-scripts/get_carter.txt' ),
#    ( 'Gothika', '../example-scripts/gothika.txt' ),
#    ( 'Groundhogs Day', '../example-scripts/groundhogs_day.txt' ),
#    ( 'Red Planet', '../example-scripts/red_planet.txt' ),
#    ( 'Smurfs', '../example-scripts/smurfs.txt' ),
#    ( 'Sweet November', '../example-scripts/sweet_november.txt' ),
#    ( 'Taking Lives', '../example-scripts/taking_lives.txt' ),
#    ( 'Thirteen Ghosts', '../example-scripts/thirteen_ghosts.txt' ),
    ]

def character_lines( script ):
    #import pdb
    #pdb.set_trace()

    name = script[0]
    script_file = script[1]

    f = open( script_file )
    raw_script = unicode( f.read(), errors = 'ignore' )

    print "Working on:", name
    file_dir = '../example-scripts/parsed/'
    file_name = re.sub( r'\s+', '_', name.lower() )
    outdir = file_dir + file_name

    #Script = tsl.script.Script.Script( name, outdir )
    #Script.load()

    Structure = tsl.script.Structure.Structure( name, outdir )
    Structure.load()

    Presences = tsl.script.Presences.Presences( name, outdir )
    Presences.load()

    structure = Structure.structure
    presence_sn = Presences.presence_sn

    total_dialog_words = structure['dialog_words']
    total_words = structure['total_words']

    scene_data = []

    locations = {}
    location_count = 0

    for scene_key in sorted( structure['scenes'].keys(), 
                         key=lambda x: structure['scenes'][x]['scene_number'] ):
        scene = structure['scenes'][scene_key]

        scene_dialog_words = scene['dialog_words']
        scene_total_words = scene['total_words']
        
        scene_location = None
        scene_characters = {}

        for ( name, presence_list ) in presence_sn[scene_key].items():
            for presence in presence_list:
                if presence['presence_type'] == SETTING:
                    scene_location = name
                    if scene_location not in locations:
                        location_count += 1
                        locations[scene_location] = location_count
                elif presence['noun_type'] == CHARACTER:
                    if name in scene_characters:
                        scene_characters[name]['appearances'] += 1
                    else:
                        scene_characters[name] = { 'dialog_words' : 0,
                                                   'dialog_percentage' : 0.0,
                                                   'appearances' : 1 }

                    if presence['presence_type'] == DISCUSS:
                        scene_characters[name]['dialog_words'] += presence['dialog_words']
                        scene_characters[name]['dialog_percentage'] = float( scene_characters[name]['dialog_words'] ) / scene_dialog_words

        scene_data.append( { 'location' : scene_location,
                             'location_number' : locations[scene_location],
                             'characters' : scene_characters,
                             'scene_dialog_words' : scene_dialog_words,
                             'scene_total_words' : scene_total_words,
                             'scene_number' : int( scene_key ),
                             'scene_percentage' : float( scene_total_words ) / total_words } )

    output = {
        'total_dialog_words' : total_dialog_words,
        'total_words' : total_words,
        'scene_data' : scene_data
        }

    f = open( file_dir + file_name + '/%s_character_lines.json' % ( file_name ), 'w' )
    json.dump( output, f, sort_keys=True, indent=4 )
    f.close()

    #print json.dumps( output, sort_keys=True, indent=4 )
            
    '''
    #Interactions = tsl.script.Interactions.Interactions( name, outdir )
    #Interactions.load()

    #output = {}

    # DEBUG - do we want to limit things to the top N chars?
    top_characters = top_presences( Presences, noun_types=[CHARACTER] )
    top_locations = top_presences( Presences, noun_types=[LOCATION] )

    # Distinct locations
    output['distinct_locations'] = len( top_locations )

    # Number of location changes.
    location_changes = 0
    current_location = None
    for presence in Presences.presences:
        if presence['presence_type'] == SETTING and presence['noun_type'] == LOCATION:
            if presence['name'] != current_location:
                location_changes += 1
            current_location = presence['name']
    output['location_changes'] = location_changes
    

    # Percentage of scenes with main character
    scenes = Presences.presence_sn.keys()
    scene_count = len( scenes )
    main_character_appearances = 0
    for scene in scenes:
        if top_characters[0][0] in Presences.presence_sn[scene]:
            main_character_appearances += 1
    output['percentage_of_scenes_with_main_character'] = float( main_character_appearances ) / scene_count
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
    
    output['title'] = name
    output['characters_speaking_in_scene_stats'] = get_stats( scene_talker_data )
    #output['characters_speaking_in_scene'] = scene_talker_data

    # Ratio of dialog to non-dialog words per scene.
    scenes = Structure.structure['scenes']
    scene_word_ratios = [ ( x[0], float( x[1]['dialog_words'] ) / x[1]['total_words'] ) for x in scenes.items() ]
    output['dialog_to_total_word_ratio_in_scenes_stats'] = get_stats( scene_word_ratios )

    # Global word counts
    output['total_words'] = Structure.structure['total_words']
    output['dialog_words'] = Structure.structure['dialog_words']

    # % of action words.
    total_action_words = 0
    for scene in Structure.structure['scenes'].keys():
        for block in Structure.structure['scenes'][scene]['scene_blocks']:
            if block['block_type'] == 'ACTION':
                total_action_words += block['total_words']
    output['total_action_words'] = total_action_words

    # % of dialog by top-N speakers.
    dialog_by_top_chars = []
    for character in top_characters:

        name = character[0]
        
        dialog = 0
        for ( scene_id, presences ) in Presences.presence_ns[name].items():
            if scene_id == 'noun_type':
                continue
            for presence in presences:
                if presence['presence_type'] == DISCUSS:
                    dialog += presence['dialog_words']
        dialog_by_top_chars.append( { 'character' : name, 'appearances' : character[2], 'percent_dialog' : float( dialog ) / Structure.structure['dialog_words'] } )

    output['percent_dialog_by_top_10_characters'] = dialog_by_top_chars[:10]
                
    # % of words in the top-N locations
    words_at_top_locs = []
    for location in top_locations:
        name = location[0]

        words = 0
        for ( scene_id, presences ) in Presences.presence_ns[name].items():
            if scene_id == 'noun_type':
                continue
            words += Structure.structure['scenes'][scene_id]['total_words']

        words_at_top_locs.append( { 'location' : name, 'appearances' : location[2], 'percent_words' : float( words ) / Structure.structure['total_words'] } )

    output['percent_words_by_top_10_locations'] = words_at_top_locs[:10]


    # Number of characters in dialog per DU.
    dus = get_dramatic_unit_partitions( Presences.presence_sn, 0.5 )
    speaker_count = []
    for du in dus:
        speakers = {}
        for scene_idx in du:
            for name, presences in Presences.presence_sn["%s" % scene_idx].items():
                if name in speakers:
                    continue
                for presence in presences:
                    if presence['noun_type'] == CHARACTER:
                        if presence['presence_type'] == DISCUSS:
                            speakers[presence['name']] = True
                            break
                    else:
                        break
        speaker_count.append( len( speakers.keys() ) )
    output['du_speakers'] = speaker_count

    # Write the output.
    f = open( file_dir + file_name + '/%s_metrics.json' % ( file_name ), 'w' )
    json.dump( output, f, sort_keys=True, indent=4 )


def get_stats( data ):
#    Input is an unsorted array of ( 'scene_id', numerical quantity
#    ) tuples.  We compute the average, min, max, median, and standard
#    deviation of the numerical quantities.
    
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
'''

def output_top_interactions( interactions, filename ):
    f = open( filename, 'w' )
    f.write("name1,name2,interactions\n")
    for i in interactions:
        f.write( ','.join( [ i[0], i[1], str( i[2] ) ] ) )
        f.write( "\n" )
    f.close()

for script in scripts:
    character_lines( script )
