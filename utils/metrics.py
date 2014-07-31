#!/usr/bin/python

import csv
import json
import math
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
    #( 'Chinatown', '../example-scripts/chinatown.txt' ),
    #( 'Dune', '../example-scripts/dune.txt' ),
    #( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    #( 'The Matrix', '../example-scripts/the_matrix.txt' ),
    #( 'Good Will Hunting', '../example-scripts/good_will_hunting.txt' ),
    #( 'The Book of Eli', '../example-scripts/the_book_of_eli.txt' ),
    #( 'Starwars', '../example-scripts/starwars.txt' ),
    #( 'Alien', '../example-scripts/alien.txt' ),
    #( 'Vertigo', '../example-scripts/vertigo.txt' ),
    #( 'Terminator 2', '../example-scripts/terminator_2.txt' ),
    #( 'Ratatouille', '../example-scripts/ratatouille.txt' ),
    # Questionable formatting
    #( 'Analyze That', '../example-scripts/analyze_that.txt' ),
    #( 'Death to Smoochy', '../example-scripts/death_to_smoochy.txt' ),
    #( 'Get Carter', '../example-scripts/get_carter.txt' ),
    #( 'Groundhogs Day', '../example-scripts/groundhogs_day.txt' ),
    #( 'Red Planet', '../example-scripts/red_planet.txt' ),
    #( 'Smurfs', '../example-scripts/smurfs.txt' ),
    #( 'Sweet November', '../example-scripts/sweet_november.txt' ),
    #( 'Taking Lives', '../example-scripts/taking_lives.txt' ),
    #( 'Thirteen Ghosts', '../example-scripts/thirteen_ghosts.txt' ),
    # New
    #( '42', '../example-scripts/42.txt' ),
    #( 'Frozen', '../example-scripts/frozen.txt' ),
    #( 'Fruitvale Station', '../example-scripts/fruitvale_station.txt' ),
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
    #( '1969 A Space Odyssey', '../example-scripts/1969_a_space_odyssey.txt' ),
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
    #( 'Frisco', '../example-scripts/frisco.txt' ),
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
    #( 'Beautiful Creatures', '../example-scripts/beautiful_creatures.txt' ),
    #( 'Section 6', '../example-scripts/section_6.txt' ),
    #( 'Seed', '../example-scripts/seed.txt' ),
    #( 'Shovel Buddies', '../example-scripts/shovel_buddies.txt' ),
    #( 'Spotlight', '../example-scripts/spotlight.txt' ),
    #( 'Superbrat', '../example-scripts/superbrat.txt' ),
    #( 'Sweetheart', '../example-scripts/sweetheart.txt' ),
    #( 'The Shark is not Working', '../example-scripts/the_shark_is_not_working.txt' ),
    #( 'The Line', '../example-scripts/the_line.txt' ),
    #( 'The Killing Floor', '../example-scripts/the_killing_floor.txt' ),
    #( 'The Fixer', '../example-scripts/the_fixer.txt' ),
    #( 'The End of the Tour', '../example-scripts/the_end_of_the_tour.txt' ),
    #( 'The Company Man', '../example-scripts/the_company_man.txt' ),
    #( 'The Boy and His Tiger', '../example-scripts/the_boy_and_his_tiger.txt' ),
    #( 'The Autopsy of Jane Doe', '../example-scripts/the_autopsy_of_jane_doe.txt' ),
    #( 'Tchaikovskys Requiem', '../example-scripts/tchaikovskys_requiem.txt' ),
    ]

def populate_release_stats( name, output ):
    released = {
        'Chinatown' : { 'rt' : .98 }, 
        'Dune' : { 'rt' : .55 }, 
        'Ghostbusters' : { 'rt' : .97 }, 
        'The Matrix' : { 'rt' : .87 }, 
        'Good Will Hunting' : { 'rt' : 0.97 }, 
        'The Book of Eli' : { 'rt' : 0.48 }, 
        'Starwars' : { 'rt' : 0.93 }, 
        'Alien' : { 'rt' : 0.97 }, 
        'Vertigo' : { 'rt' : 0.98 }, 
        'Terminator 2' : { 'rt' : 0.92 }, 
        'Ratatouille' : { 'rt' : 0.96 }, 
        'Analyze That' : { 'rt' : 0.27 }, 
        'Batman Begins' : { 'rt' : 0.85 }, 
        'Death to Smoochy' : { 'rt' : 0.42 }, 
        'Get Carter' : { 'rt' : 0.12 }, 
        'Gothika' : { 'rt' : 0.15 }, 
        'Groundhogs Day' : { 'rt' : 0.97 }, 
        'Red Planet' : { 'rt' : 0.14 }, 
        'Smurfs' : { 'rt' : 0.22 }, 
        'Sweet November' : { 'rt' : 0.16 }, 
        'Taking Lives' : { 'rt' : 0.22 }, 
        'Thirteen Ghosts' : { 'rt' : 0.14 }, 
        '42' : { 'rt' : 0.79 }, 
        'Frozen' : { 'rt' : 0.89 }, 
        'Fruitvale Station' : { 'rt' : 0.93 }, 
        'All is Lost' : { 'rt' : 0.93 }, 
        'Amour' : { 'rt' : 0.93 }, 
        'Argo' : { 'rt' : 0.96 }, 
        'August Osage County' : { 'rt' : 0.64 }, 
        'Celest and Jesse Forever' : { 'rt' : 0.7 }, 
        'Chronicle' : { 'rt' : 0.85 }, 
        'Dallas Buyers Club' : { 'rt' : 0.93 }, 
        'Despicable Me 2' : { 'rt' : 0.74 }, 
        'The Wolf of Wall Street' : { 'rt' : .76 }, 
        'Prince of Persia' : { 'rt' : 0.36 }, 
        'Oz the Great and Powerful' : { 'rt' : 0.59 }, 
        'Nebraska' : { 'rt' : 0.92 }, 
        'Monsters University' : { 'rt' : 0.78 }, 
        'Magic Mike' : { 'rt' : 0.8 }, 
        'Lone Survivor' : { 'rt' : 0.75 }, 
        'Kill Your Darlings' : { 'rt' : 0.76 }, 
        'Kick Ass 2' : { 'rt' : 0.29 }, 
        'The Great Gatsby' : { 'rt' : 0.49 }, 
        'The Invisible Woman' : { 'rt' : 0.76 }, 
        'The Past' : { 'rt' : 0.94 }, 
        'Twilight' : { 'rt' : 0.49 }, 
        'Wadjda' : { 'rt' : 0.99 }, 
        'Woman in Black' : { 'rt' : 0.66 }, 
        'Prisoners' : { 'rt' : 0.82 }, 
        'Real Steel' : { 'rt' : 0.6 }, 
        'Rush' : { 'rt' : 0.89 }, 
        'Rust and Bone' : { 'rt' : 0.82 }, 
        'Skyfall' : { 'rt' : 0.92 }, 
        'Smashed' : { 'rt' : 0.84 }, 
        'Snow White and the Huntsman' : { 'rt' : 0.48 }, 
        'The Croods' : { 'rt' : 0.7 }, 
        'Beautiful Creatures' : { 'rt' : 0.46 },
        }

    if name not in released:
        #raise Exception( "ERROR - NO DATA FOR %s" % ( name ) )
        pass
    else:
        output['rt'] = released[name]['rt']

    return

def process_script( script ):
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

    Script = tsl.script.Script.Script( name, outdir )
    Script.load()

    Structure = tsl.script.Structure.Structure( name, outdir )
    Structure.load()

    Presences = tsl.script.Presences.Presences( name, outdir )
    Presences.load()

    Interactions = tsl.script.Interactions.Interactions( name, outdir )
    Interactions.load()

    output = {}

    populate_release_stats( name, output )

    output['title'] = name

    top_characters = top_presences( Presences, noun_types=[CHARACTER] )

    output['named_characters'] = len( top_characters )

    top_locations = top_presences( Presences, noun_types=[LOCATION] )

    # Collocations
    if False:
        all_words = nltk.wordpunct_tokenize( raw_script )
        word_tokens = [ t.lower() for t in all_words if re.search( r'\w', t ) ]
        text = nltk.Text( word_tokens )
        print text.collocations()

    # Distinct words.
    #import pdb
    #pdb.set_trace()
        
    all_words = nltk.wordpunct_tokenize( raw_script )
    word_tokens = [ t for t in all_words if re.search( r'\w', t ) ]
    text = nltk.Text( word_tokens )
    words = [ w.lower() for w in text ]
    vocab = sorted( set( words ) )
    english_words = set( w.lower() for w in nltk.corpus.words.words() )
    exotic_words = [ e for e in words if e not in english_words ]
    distinct_exotic_words = set( exotic_words )
    #output['exotic_words'] = len( exotic_words )
    #output['distinct_exotic_words'] = len( distinct_exotic_words )
    output['distinct_words'] = len( vocab )

    # Word categories
    if True:
        tagged_words = nltk.pos_tag( text )
        porter = nltk.PorterStemmer()
        stemmed_tags = [ ( porter.stem( t[0].lower() ), t[1] ) for t in tagged_words ]
        fd = nltk.FreqDist( stemmed_tags )
        verbs = [ word for ( word, tag ) in fd if ( tag.startswith('V') and len( word ) > 3 )]
        adj = [ word for ( word, tag ) in fd if ( tag.startswith('JJ') and len( word ) > 1 ) ]
        adv = [ word for ( word, tag ) in fd if ( tag.startswith('RB') and len( word ) > 1 ) ]
        nouns = [ word for ( word, tag ) in fd if ( tag in ('NN', 'NNS' ) and len( word ) > 1 ) ]
           
        output['adj-adv_noun-verb_ratio'] = float( ( len( adj ) + len( adv ) ) ) / ( len( nouns ) + len( verbs ) )
      
        #output['vocabulary'] = {}
        #output['vocabulary']['adjectives'] = adj[:10]
        #output['vocabulary']['non_proper_nouns'] = nouns[:10]
        #output['vocabulary']['verbs'] = verbs[:10]

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

        stemmed_tags.dispersion_plot( [ word[0] for word in stemmed_tags if word[1].startswith( 'V' ) ] )

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

    # Number of characters who speak to the N'th speaker.
    character_x_speakers = []
    #interlocutor_score = 0
    for character in top_characters[:5]:
        name = character[0]
        speakers = top_interactions( Presences, Interactions, noun_types=[( CHARACTER, CHARACTER )], interaction_types=[DISCUSS], names=[name] ) 
        if name == top_characters[0][0]:
            character_x_speakers.append( { 'character' : name, 'speakers' : 3*len( speakers ) } )
            #interlocutor_score += 3* len( speakers )
        else:
            character_x_speakers.append( { 'character' : name, 'speakers' : len( speakers ) } )
            #interlocutor_score += len( speakers )
    output['character_x_speakers'] = character_x_speakers
    #output['interlocutor_score'] = interlocutor_score

    #output['main_character_interlocutor_count'] = len( speakers )
    
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
    output['location_changes'] = math.log( location_changes )
    

    # Percentage of scenes with main character
    scenes = Presences.presence_sn.keys()
    scene_count = len( scenes )
    main_character_appearances = 0
    for scene in scenes:
        if top_characters[0][0] in Presences.presence_sn[scene]:
            main_character_appearances += 1
    #output['percentage_of_scenes_with_main_character'] = float( main_character_appearances ) / scene_count
    #output['main_character'] = top_characters[0][0]

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
    
    #output['characters_speaking_in_scene_stats'] = get_stats( scene_talker_data )
    scene_talker_stats = get_stats( scene_talker_data )
    output['scene_dialog_score'] = scene_talker_stats['median']

    #output['characters_speaking_in_scene'] = scene_talker_data

    # Ratio of dialog to non-dialog words per scene.
    scenes = Structure.structure['scenes']
    scene_word_ratios = [ ( x[0], float( x[1]['dialog_words'] ) / x[1]['total_words'] ) for x in scenes.items() ]
    #output['dialog_to_total_word_ratio_in_scenes_stats'] = get_stats( scene_word_ratios )

    # Global word counts
    #output['total_words'] = Structure.structure['total_words']
    #output['dialog_words'] = Structure.structure['dialog_words']
    output['percent_dialog'] = float( Structure.structure['dialog_words'] ) / Structure.structure['total_words']

    # Percentage of dialog in the 1st, 2nd, Nth portion of the text.
    nths = 4;
    nth_percent_of_dialog = []
    percent_of_dialog_in_nth = []
    for i in range( nths ):
        nth_percent_of_dialog.append( 0 )
        percent_of_dialog_in_nth.append( 0 )

    # total_words, dialog_words
    total_words = Structure.structure['total_words']
    total_dialog = Structure.structure['dialog_words']
    running_word_count = 0
    current_nth = 0
    nth_word_count = 0
    nth_dialog_count = 0
    for scene_id in sorted( Structure.structure['scenes'].keys(), key=int ):
        scene = Structure.structure['scenes'][scene_id]
        nth_word_count += scene['total_words']
        nth_dialog_count += scene['dialog_words']
        running_word_count += scene['total_words']
        new_nth = int( nths * running_word_count / total_words )
        if new_nth != current_nth:
            nth_percent_of_dialog[current_nth] = float( nth_dialog_count ) / total_dialog
            percent_of_dialog_in_nth[current_nth] = float( nth_dialog_count ) / nth_word_count
            current_nth = new_nth
            nth_word_count = 0
            nth_dialog_count = 0

    #output['nth_percent_of_dialog'] = nth_percent_of_dialog
    #output['percent_of_dialog_in_nth'] = percent_of_dialog_in_nth

    # % of action words.
    total_action_words = 0
    for scene in Structure.structure['scenes'].keys():
        for block in Structure.structure['scenes'][scene]['scene_blocks']:
            if block['block_type'] == 'ACTION':
                total_action_words += block['total_words']
    #output['total_action_words'] = total_action_words

    # % of dialog by top-N speakers.
    dialog_by_top_chars = []
    #dialog_score = 0
    for character in top_characters[:5]:
        name = character[0]
        
        dialog = 0
        for ( scene_id, presences ) in Presences.presence_ns[name].items():
            if scene_id == 'noun_type':
                continue
            for presence in presences:
                if presence['presence_type'] == DISCUSS:
                    dialog += presence['dialog_words']
        #dialog_by_top_chars.append( { 'character' : name, 'appearances' : character[2], 'percent_dialog' : float( dialog ) / Structure.structure['dialog_words'] } )
        if name == top_characters[0][0]:
            dialog_by_top_chars.append( { 'character' : name, 'appearances' : character[2], 'percent_dialog' : 3 * float( dialog ) / Structure.structure['dialog_words'] } )
            #dialog_score += 3*dialog_by_top_chars[-1]['percent_dialog']
        else:
            dialog_by_top_chars.append( { 'character' : name, 'appearances' : character[2], 'percent_dialog' : float( dialog ) / Structure.structure['dialog_words'] } )
            #dialog_score += dialog_by_top_chars[-1]['percent_dialog']
            
    output['percent_dialog_by_character'] = dialog_by_top_chars
    #output['dialog_score'] = dialog_score

    # Number of non-main characters who have at least 15 words of
    # dialog.
    supporting_characters = 0
    for character in top_characters[1:]:
        name = character[0]
        
        dialog = 0
        for ( scene_id, presences ) in Presences.presence_ns[name].items():
            if scene_id == 'noun_type':
                continue
            for presence in presences:
                if presence['presence_type'] == DISCUSS:
                    dialog += presence['dialog_words']
        if dialog >= 15:
            supporting_characters += 1

    output['supporting_characters'] = math.log( supporting_characters )
           
    # % of scenes that have the top-N characters.
    scenes_with_top_chars = []
    #presence_score = 0
    for character in top_characters[:5]:
        name = character[0]
        scenes = len( Presences.presence_sn.keys() )
        count = 0
        for scene in Presences.presence_sn.keys():
            if name in Presences.presence_sn[scene]:
                count += 1
        #scenes_with_top_chars.append( { 'character' : name, 'percentage_of_scenes' : float( count ) / scenes } )
        if name == top_characters[0][0]:
            scenes_with_top_chars.append( { 'character' : name, 'percentage_of_scenes' : 3 * float( count ) / scenes } )
            #presence_score += 3*scenes_with_top_chars[-1]['percentage_of_scenes']
        else:
            scenes_with_top_chars.append( { 'character' : name, 'percentage_of_scenes' : 3 * float( count ) / scenes } )
            #presence_score += scenes_with_top_chars[-1]['percentage_of_scenes']

    #output['presence_score'] = presence_score
    output['scenes_percentage_for_characters'] = scenes_with_top_chars

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

    #output['percent_words_by_top_10_locations'] = words_at_top_locs[:10]

    # Number of characters in dialog per DU.
    dus = get_dramatic_unit_partitions( Presences.presence_sn, 0.5 )
    output['dramatic_units'] = len( dus )
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
    #output['du_speakers'] = speaker_count

    # Stats on number of words per unit of dialog.
    dialog_stats = []
    for scene in Structure.structure['scenes'].keys():
        for block in Structure.structure['scenes'][scene]['scene_blocks']:
            if block['block_type'] == 'DIALOG':
                dialog_stats.append( ( scene, block['total_words'] ) )
    #output['dialog_block_word_count_stats'] = get_stats( dialog_stats )
    dialog_score_stats = get_stats( dialog_stats )
    output['dialog_words_score'] = [ dialog_score_stats['average'],  dialog_score_stats['max'] ]
   
    # Hearing - sum of number of characters speaking in a scene *
    # words of dialog in a scene.
    hearing = 0
    for scene in Structure.structure['scenes'].keys():
        scene_dialog = Structure.structure['scenes'][scene]['dialog_words']
        if scene_dialog > 0:
            scene_presences = Presences.presence_sn[scene]
            speakers = 0
            for person in scene_presences.keys():
                presence_list = scene_presences[person]
                for presence in presence_list:
                    if presence['presence_type'] == DISCUSS:
                        speakers += 1
                        break
            hearing += scene_dialog * speakers
    output['hearing'] = hearing

    # Buddies - the number of characters pairs a, b such that whenever
    # a appears b is present at least 50% of the time.
    buddies = []
    buddy_threshold = 0.5
    min_appearances = 3
    for character in sorted( top_characters ):
        name = character[0]
        character_scene_ids = [ x for x in Presences.presence_ns[name].keys() if x != 'noun_type' ]
        if len( character_scene_ids ) <= min_appearances:
            continue
        for co_character in sorted( top_characters ):
            co_name = co_character[0]
            if co_name <= name:
                continue
            co_character_scene_ids = [ x for x in Presences.presence_ns[co_name].keys() if x != 'noun_type' ]
            if len( co_character_scene_ids ) <= min_appearances:
                continue
            ratio_a = float( len( set( character_scene_ids ).intersection( set( co_character_scene_ids ) ) ) ) / len( character_scene_ids )
            ratio_b = float( len( set( character_scene_ids ).intersection( set( co_character_scene_ids ) ) ) ) / len( co_character_scene_ids )
            if ratio_a >= buddy_threshold and ratio_b >= buddy_threshold:
                buddies.append( ( name, co_name, ratio_a, ratio_b ) )
    #output['buddies_list'] = buddies
    #output['buddies'] = len( buddies )

    # Write the output.
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

f = open( "/wintmp/movies.csv", 'r' )
movie_csv = csv.reader( f )

release_stats = {}

#for row in movie_csv:
#    if row[6] in release_stats:
#        print "ERROR - SECOND COPY OF:", row[6]
#    release_stats[row[6]] = {
#        'date' : row[0],
#        'domestic_box' : row[1],
#        'international_box' : row[2],
#        'domestic_dvd_sales' : row[3],
#        'production_budget' : row[4],
#        'rt' : row[5],
#        'name' : row[6],
#        'key' : row[7],
#        }

#import pdb
#pdb.set_trace()

for script in scripts:
    process_script( script )

