#!/usr/bin/python

import json
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy
import operator
import re
import scipy
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
import sys
import uuid

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
#    ( 'The Matrix', '../example-scripts/the_matrix.txt' ),
#    ( 'Good Will Hunting', '../example-scripts/good_will_hunting.txt' ),
#    ( 'The Book of Eli', '../example-scripts/the_book_of_eli.txt' ),
#    ( 'Starwars', '../example-scripts/starwars.txt' ),
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

    # Nodes are a list of name:string pairs
    #
    # Links are a list of source, target, value triples where source
    # and target are the indices of the nodes in the nodes array.
    sankey = { 'nodes' : [],
               'links' : [] }

    # Integer character value with the integer location value where
    # this character was last seen.
    sankey_char_last_seen = {}

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

    top_ns = [ 1, 2, 4, 8, 16, 1024 ]
    top_ns = [ 2 ] 

    for top_n in top_ns:
        scene_data = []

        locations = {}
        location_count = 0
        running_words = 0

        viz = nx.Graph()
        viz_positions = {}
        viz_labels = {}
        viz_pending_edges = {}
        # Hash of [ total words, first half words, second half words ]
        setting_appearances = {}

        top_characters = top_presences( Presences, noun_types=[CHARACTER], top_n=top_n )
        top_character_names = {}
        character_names_to_number = {}
        character_number = 0
        
        co_occurences = numpy.zeros( shape=( top_n, top_n ) )

        for character in top_characters:
            top_character_names[character[0]] = True
            character_names_to_number[character[0]] = character_number
            co_occurences[character_number][character_number] = 1
            
            sankey['nodes'].append( { 'name' : character[0] } )

            character_number += 1

        for scene_key in sorted( structure['scenes'].keys(), 
                                 key=lambda x: structure['scenes'][x]['scene_number'] ):
            scene = structure['scenes'][scene_key]

            scene_dialog_words = scene['dialog_words']
            scene_total_words = scene['total_words']

            scene_location = None
            scene_characters = {}
            scene_character_list = []

            scene_uuid = str( uuid.uuid4() )

            for ( name, presence_list ) in presence_sn[scene_key].items():
                for presence in presence_list:
                    if presence['presence_type'] == SETTING:
                        scene_location = name
                        if scene_location not in locations:
                            location_count += 1
                            locations[scene_location] = location_count

                        sankey['nodes'].append( { 'name' : scene_location } )

                        viz.add_node( scene_uuid )
                        viz_positions[scene_uuid] = [ 100*float( running_words ) / total_words, locations[scene_location] ]
                        viz_labels[scene_uuid] = scene_location
                    
                    elif presence['noun_type'] == CHARACTER and name in top_character_names:
                        if name in scene_characters:
                            scene_characters[name]['appearances'] += 1
                        else:
                            scene_characters[name] = { 'dialog_words' : 0,
                                                       'dialog_percentage' : 0.0,
                                                       'appearances' : 1 }
                            scene_character_list.append( character_names_to_number[name] )

                        if presence['presence_type'] == DISCUSS:
                            scene_characters[name]['dialog_words'] += presence['dialog_words']
                            scene_characters[name]['dialog_percentage'] = float( scene_characters[name]['dialog_words'] ) / scene_dialog_words
                        
                        if name in viz_pending_edges.keys():
                            viz.add_edge( scene_uuid, viz_pending_edges[name] )
                            viz_pending_edges[name] = scene_uuid
                        else:
                            viz_pending_edges[name] = scene_uuid

            if scene_location not in setting_appearances:
                setting_appearances[scene_location] = [0,0,0]

            setting_appearances[scene_location][0] += scene_total_words
            if running_words < float( total_words ) / 2:
                setting_appearances[scene_location][1] += scene_total_words
            else:
                setting_appearances[scene_location][2] += scene_total_words

            running_words += scene_total_words

            for i in scene_character_list:
                for j in scene_character_list:
                    co_occurences[i][j] += 1
                    if i != j:
                        co_occurences[j][i] += 1

            # Sankey links.
            for i in scene_character_list:
                current_loc = len( scene_data )
                if i not in sankey_char_last_seen:
                    sankey['links'].append( { 'source' : i, 'target' : current_loc, 'value' : 1 } )
                elif sankey_char_last_seen[i] != current_loc:
                    sankey['links'].append( { 'source' : sankey_char_last_seen[i], 'target' : current_loc, 'value' : 1 } )

                sankey_char_last_seen[i] = current_loc

            scene_data.append( { 'duration' : scene_total_words,
                                 'start' : running_words - scene_total_words,
                                 'id' : int( scene_key ) - 1,
                                 'chars' : scene_character_list,
                                 'location' : scene_location,
                                 'location_number' : locations[scene_location],
                                 'characters' : scene_characters,
                                 'scene_dialog_words' : scene_dialog_words,
                                 'scene_total_words' : scene_total_words,
                                 'scene_number' : int( scene_key ),
                                 'scene_start_percentage' : float( running_words ) / total_words,
                                 'scene_percentage' : float( scene_total_words ) / total_words } )

        # Refactor the y coordinates of settings.
        # Extract the setting with the most words:
        max_setting_key = max( setting_appearances.iteritems(), key=lambda x: x[1][0] )[0]
        max_setting = setting_appearances[max_setting_key]
        setting_appearances.pop( max_setting_key )
        
        first_half_settings = [ x for x in setting_appearances.iteritems() if x[1][1] > x[1][2] ]
        second_half_settings = [ x for x in setting_appearances.iteritems() if x[1][1] <= x[1][2] ]

        first_half_settings = sorted( first_half_settings, key=lambda x: x[1][0] )
        second_half_settings = sorted( second_half_settings, key=lambda x: x[1][0] )
        second_half_settings.reverse()

        ordered_settings = first_half_settings + [ ( max_setting_key, max_setting ) ] + second_half_settings
        setting_numbers = {}

        for ( idx, setting ) in enumerate( ordered_settings ):
            setting_numbers[setting[0]] = idx
            
        for key in viz_positions.keys():
            viz_positions[key][1] = setting_numbers[viz_labels[key]]

        plt.figure( 1, figsize=( 60, 40 ) )
        nx.draw( viz, pos=viz_positions, labels=viz_labels, font_size=8 ) #with_labels=False )
        plt.axis( [ -1, 101, -1, 110 ] )
        plt.savefig( '/wintmp/movie/character_lines/%s_top_%s.png' % ( file_name, top_n ), dpi=100 )
        plt.close()

        output = {
            'total_dialog_words' : total_dialog_words,
            'total_words' : total_words,
            'scenes' : scene_data,
            'panels' : total_words
            }

        f = open( file_dir + file_name + '/%s_top_%s_character_lines.json' % ( file_name, top_n ), 'w' )
        json.dump( output, f, sort_keys=True, indent=4 )
        f.close()


        normalized = normalize( co_occurences )
        distances = sklearn.metrics.pairwise.pairwise_distances( normalized )
        groups = DBSCAN( min_samples=1 ).fit_predict( distances )

        char_file = open( file_dir + file_name + '/%s_top_%s_characters.xml' % ( file_name, top_n ), 'w' )
        char_file.write( "<characters>\n" )

        for ( idx, character ) in enumerate( top_characters ):
            char_file.write( '<character group="%s" id="%s" name="%s" />\n' % ( int( groups[idx] ), idx, character[0].lower().capitalize() ) )

        char_file.write( "</characters>\n" )
        char_file.close()

        #sankey_file = open( "sankey.json", 'w' )
        #json.dump( sankey, sankey_file, indent=4 )
        #sankey_file.close()
    #print json.dumps( output, sort_keys=True, indent=4 )
            
def output_top_interactions( interactions, filename ):
    f = open( filename, 'w' )
    f.write("name1,name2,interactions\n")
    for i in interactions:
        f.write( ','.join( [ i[0], i[1], str( i[2] ) ] ) )
        f.write( "\n" )
    f.close()

for script in scripts:
    character_lines( script )
