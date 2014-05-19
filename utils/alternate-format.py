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

from tsl.script.parse.const import CHARACTER, DISCUSS, LOCATION, SETTING, ACTION, DIALOG, DIALOG_HEADER
from tsl.script.reports.reports import top_presences, top_interactions, get_presence_csv, get_interaction_csv

from tsl.utils.partition import get_dramatic_unit_partitions

scripts = [
    #( 'Chinatown', '../example-scripts/chinatown.txt' ),
    #( 'Dune', '../example-scripts/dune.txt' ),
    # ( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    #( 'Recursion', '../example-scripts/recursion-05-16-14.txt' ),
    ( 'Starthur', '../example-scripts/starthur.txt' ),
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
    #( 'Batman Begins', '../example-scripts/batman_begins.txt' ),
    #( 'Death to Smoochy', '../example-scripts/death_to_smoochy.txt' ),
    #( 'Get Carter', '../example-scripts/get_carter.txt' ),
    #( 'Gothika', '../example-scripts/gothika.txt' ),
    #( 'Groundhogs Day', '../example-scripts/groundhogs_day.txt' ),
    #( 'Red Planet', '../example-scripts/red_planet.txt' ),
    #( 'Smurfs', '../example-scripts/smurfs.txt' ),
    #( 'Sweet November', '../example-scripts/sweet_november.txt' ),
    #( 'Taking Lives', '../example-scripts/taking_lives.txt' ),
    #( 'Thirteen Ghosts', '../example-scripts/thirteen_ghosts.txt' ),
    # New
    #( '42', '../example-scripts/42.txt' )
    #( 'Frozen', '../example-scripts/frozen.txt' )
    #( 'Fruitvale Station', '../example-scripts/fruitvale_station.txt' )
    ]

author = "Ben Chelf"
email = "bchelf@thestorylocker.com"
rating = "LIKE"

def get_lines_per_page( Script ):
    result = {}

    current_page = Script.script_lines[0]['page_no']
    first_line = Script.script_lines[0]['line_no']
    last_line = None

    for line in Script.script_lines:
        this_page = line['page_no']
        this_line = line['line_no']

        if this_page != current_page:
            result[current_page] = { "first_line" : first_line,
                                     "last_line"  : this_line - 1,
                                     "lines" : this_line - first_line }
            first_line = this_line
            current_page = this_page
            
        last_line = this_line

    result[current_page] = { "first_line" : first_line,
                             "last_line"  : this_line - 1,
                             "lines" : this_line - first_line }

    return result

def get_timestamp( page, line, lines_per_page, first_page ):
    lines = lines_per_page[page]['lines']
    return int( 60000 * ( page - first_page + float( line - lines_per_page[page]['first_line'] ) / lines ) )
    
def handle_block_type( block_type, block, Script, Structure, lines_per_page, first_page ):
    
    first_line = block['first_line']
    last_line = block['last_line']
    page = Script.script_lines[first_line - 1]['page_no']

    timestamp = get_timestamp( page, first_line, lines_per_page, first_page )

    content = "".join( [ x['content'] for x in Script.script_lines[first_line-1:last_line] ] )

    return {
        "author"     : { "username" : email },
        "content"    : content,
        "happens_at" : timestamp,
        "rating"     : rating,
        "type"       : block_type
        }

def handle_action( block, Script, Structure, lines_per_page, first_page ):
    return handle_block_type( "PLOT", block, Script, Structure, lines_per_page, first_page )

def get_speaker_blocks( block ):
    current_line_type = None
    prior_block = None

    result = []

    for line_type_key in sorted( block['line_types'].keys(), key=int ):
        line_no = int( line_type_key )
        line_type = block['line_types'][line_type_key]
        
        if line_type == DIALOG_HEADER:
            if prior_block is not None:
                prior_block["last_line"] = line_no - 1
                result.append( prior_block )

            prior_block = { "first_line" : line_no }

    if prior_block is not None and line_type is not DIALOG_HEADER:
        prior_block['last_line'] = line_no
        result.append( prior_block )

    return result
        
def handle_dialog( block, Script, Structure, lines_per_page, first_page ):
    speaker_blocks = get_speaker_blocks( block )

    result = []
    
    for speaker_block in speaker_blocks:
        result.append( handle_block_type( "CHARACTER", speaker_block, Script, Structure, lines_per_page, first_page ) )

    return result

'''
    "storyelements": [{
      "author": {
        "username": "bchelf@thestorylocker.com"
      }, 
      "content": "Looks at closet?\n", 
      "happens_at": 9677, 
      "rating": "LIKE", 
      "type": "PLOT"
    }]



<storyelement>
  <author>
    <name>Ben Chelf</name>
    <username>bchelf@thestorylocker.com</username>
  </author>
  <content> ACTION CHUNK </content>
  <happens_at> MILLISECOND CALCULATION* (see below) </happens_at>
  <rating>LIKE</rating>
  <type>PLOT</type>
</storyelement>

and for each line of Dialog chunk:
<storyelement>
  <author>
    <name>Ben Chelf</name>
    <username>bchelf@thestorylocker.com</username>
  </author>
  <content>SPEAKER NAME: DIALOG CHUNK </content>
  <happens_at> MILLISECOND CALCULATION* (see below) </happens_at>
  <rating>LIKE</rating>
  <type>CHARACTER</type>
</storyelement>
'''

def process_script( script ):
    #import pdb
    #pdb.set_trace()

    name = script[0]
    script_file = script[1]

    f = open( script_file )
    raw_script = unicode( f.read(), errors = 'ignore' )

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

    first_line = Structure.structure['scenes']['1']['first_line']
    first_page = Script.script_lines[int( first_line ) - 1]['page_no']

    output = []

    lines_per_page = get_lines_per_page( Script )

    for scene_key in sorted( Structure.structure['scenes'].keys(), key=int ):
        scene = Structure.structure['scenes'][scene_key]

        for block in scene['scene_blocks']:
            result = None
            if block['block_type'] == ACTION:
                result = [ handle_action( block, Script, Structure, lines_per_page, first_page ) ]
            elif block['block_type'] == DIALOG:
                result = handle_dialog( block, Script, Structure, lines_per_page, first_page )
                
            if result is not None:
                output += result

    print json.dumps( output, sort_keys=True, indent=2 )

            
for script in scripts:
    process_script( script )
