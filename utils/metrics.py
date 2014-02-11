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
    ( 'Chinatown', '../example-scripts/chinatown.txt' ),
    ( 'Dune', '../example-scripts/dune.txt' ),
    ( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    ( 'The Matrix', '../example-scripts/the_matrix.txt' ),
    ( 'Good Will Hunting', '../example-scripts/good_will_hunting.txt' ),
    ( 'The Book of Eli', '../example-scripts/the_book_of_eli.txt' ),
    ( 'Starwars', '../example-scripts/starwars.txt' ),
    ( 'Alien', '../example-scripts/alien.txt' ),
    ( 'Vertigo', '../example-scripts/vertigo.txt' ),
    ( 'Terminator 2', '../example-scripts/terminator_2.txt' )
    ]

def process_script( script ):
    #import pdb
    #pdb.set_trace()

    name = script[0]
    script_file = script[1]

    f = open( script_file )
    raw_script = f.read()

    print "Working on:", name

    file_dir = '../example-scripts/parsed/'
    file_name = re.sub( r'\s+', '_', name.lower() )

    outdir = file_dir + file_name

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

    # Collocations
    if False:
        all_words = nltk.wordpunct_tokenize( raw_script )
        word_tokens = [ t.lower() for t in all_words if re.search( r'\w', t ) ]
        text = nltk.Text( word_tokens )
        print text.collocations()

    # Distinct words.
    all_words = nltk.wordpunct_tokenize( raw_script )
    word_tokens = [ t for t in all_words if re.search( r'\w', t ) ]
    text = nltk.Text( word_tokens )
    words = [ w.lower() for w in text ]
    vocab = sorted( set( words ) )
    output['distinct_words'] = len( vocab )

    # Word categories
    if False:
        tagged_words = nltk.pos_tag( text )
        porter = nltk.PorterStemmer()
        stemmed_tags = [ ( porter.stem( t[0].lower() ), t[1] ) for t in tagged_words ]
        fd = nltk.FreqDist( stemmed_tags )
        verbs = [ word for ( word, tag ) in fd if ( tag.startswith('V') and len( word ) > 3 )]
        adj = [ word for ( word, tag ) in fd if ( tag.startswith('JJ') and len( word ) > 1 ) ]
        adv = [ word for ( word, tag ) in fd if ( tag.startswith('RB') and len( word ) > 1 ) ]
        nouns = [ word for ( word, tag ) in fd if ( tag in ('NN', 'NNS' ) and len( word ) > 1 ) ]
                 
        output['vocabulary'] = {}
        output['vocabulary']['adjectives'] = adj[:10]
        output['vocabulary']['non_proper_nouns'] = nouns[:10]
        output['vocabulary']['verbs'] = verbs[:10]

    # Word categories
    #
    # DEBUG - Limit this to action text.
    if False:
        tagged_words = nltk.pos_tag( text )
        porter = nltk.PorterStemmer()
        stemmed_tags = [ ( porter.stem( t[0].lower() ), t[1] ) for t in tagged_words ]

        import pylab

        text_to_comp = stemmed_tags

        #print text_to_comp

        verbs = [ text_to_comp[x][0] for x in range( len( text_to_comp ) )
                  if text_to_comp[x][1].startswith( 'V' ) and len( text_to_comp[x][0] ) > 2 and text_to_comp[x][0] not in [ 'think', 'know', 'want', 'feel', 'see', 'here', 'say', 'happen', 'move', 'touch', 'there', 'have', 'were', 'are', 'saw', 'was', 'said', 'thought', 'knew', 'wanted', 'felt', 'happened', 'moved', 'touched', 'had', 'gitt' ] ]

        print verbs

        points = [(x,y) for x in range( len( text_to_comp ) )
                  for y in range( 1 )
                  if text_to_comp[x][1].startswith( 'V' ) and len( text_to_comp[x][0] ) > 2 and text_to_comp[x][0] not in [ 'think', 'know', 'want', 'feel', 'see', 'here', 'say', 'happen', 'move', 'touch', 'there', 'have', 'were', 'are', 'saw', 'was', 'said', 'thought', 'knew', 'wanted', 'felt', 'happened', 'moved', 'touched', 'had', 'gitt' ] ]

        if points:
            x, y = list(zip(*points))
        else:
            x = y = ()
        pylab.plot(x, y, "b|", scalex=.1)
        pylab.yticks(list(range(1)), ['verbs'], color="b")
        pylab.ylim(-1, 1)
        pylab.title("Lexical Dispersion Plot")
        pylab.xlabel("Word Offset")
        pylab.show()

        #stemmed_tags.dispersion_plot( [ word[0] for word in stemmed_tags if word[1].startswith( 'V' ) ] )

        '''
        for idx, tword in enumerate( stemmed_tags ):
            word, tag = tword
            if not tag.startswith( 'V' ) or len( word ) <= 2:
                continue
            if word not in [ 'think', 'know', 'want', 'feel', 'see', 'here', 'say', 'happen', 'move', 'touch', 'there', 'have' ]:
                print idx
                '''
        sys.exit( 0 )



    # R-numbers
    if False:
        porter = nltk.PorterStemmer()
        word_line_offsets = {}
        for line in Script.script_lines:
            content = line['content']
            line_no = line['line_no']

            line_words = nltk.wordpunct_tokenize( content )
            line_stems = [ porter.stem( w.lower() ) for w in line_words if re.search( r'\w', w ) ]

            for stem in line_stems:
                if stem in word_line_offsets:
                    word_line_offsets[stem].append( line_no )
                else:
                    word_line_offsets[stem] = [ line_no ]


        lines_per_page = 56
        print "word, occurences, max_page_gap, num_gaps_15_page_or_more, num_gaps_30_page_or_more"
        for word in sorted( word_line_offsets.keys() ):
            offsets = word_line_offsets[word]
            gap_lengths =  [ ( offsets[idx] - offsets[max(idx-1, 0)] ) for idx in range( len( offsets ) ) ]
            max_gap = max( gap_lengths )
        
            num_15_pagegap = sum( [ 1 for gap in gap_lengths if gap >= lines_per_page*15 ] )
            num_30_pagegap = sum( [ 1 for gap in gap_lengths if gap >= lines_per_page*30 ] )

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

    f = open( file_dir + file_name + '/%s_metrics.json' % ( file_name ), 'w' )
    json.dump( output, f, sort_keys=True, indent=4 )


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
