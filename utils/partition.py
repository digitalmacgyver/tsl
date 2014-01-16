#!/usr/bin/env python

import re

import tsl.script.Presences
import tsl.script.Script
import tsl.script.Structure

scripts = [
    ( 'The Big Lebowski', '../example-scripts/the_big_lebowski.txt' ),
    ( 'Chinatown', '../example-scripts/chinatown.txt' ),
    ( 'Dune', '../example-scripts/dune.txt' ),
    ( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    ( 'The Matrix', '../example-scripts/the_matrix.txt' ),
    ]

def get_partitions( presence, c ):
    prior_chars = set()
    current_chars = set()

    partitions = []
    scenes = []

    for scene in sorted( presence.keys(), key=int ):
        '''If he current set is empty, don't change definition of
        prior, include this in set.'''

        current_chars = set( [ x[0] for x in presence[scene].items() if x[1][0]['noun_type'] == "CHARACTER" ] )

        if len( current_chars ) == 0:
            #print "Working on %s" % scene
            #print current_chars
            scenes.append( scene )
        else:
            total_chars = len( current_chars | prior_chars )
            common_chars = len( current_chars & prior_chars )

            #print "Working on %s" % scene
            #print current_chars
            #print "There are %s current, %s prior, %s total, %s common characters." % ( len( current_chars ), len( prior_chars ), total_chars, common_chars )
        
            if total_chars == 0 or common_chars / float( total_chars ) >= c:
                scenes.append( scene )
            else:
                partitions.append( scenes )
                scenes = [ scene ]

            prior_chars = current_chars
        
    partitions.append( scenes )

    return partitions[1:]

def print_script( structure, text, name, partitions, c ):
    with open( '/wintmp/script-partitions/%s-%s.txt' % ( name, c ), 'wb' ) as outfile:
        for idx, p in enumerate( partitions ):
            outfile.write( "=============================== PARTITION %03d ===============================\n" % idx )
            for s in p:
                first_line = structure['scenes'][s]['first_line']
                last_line = structure['scenes'][s]['last_line']
                for line in text[ int( first_line )-1 : int( last_line )-2 ]:
                    outfile.write( line['content'] )


coefs = [ 0.01, 0.25 , 0.5, 1 ]

for film in scripts:
    name = film[0]
    
    outdir = '../example-scripts/parsed/' + re.sub( r'\s+', '_', name.lower() )

    Script = tsl.script.Script.Script( name, outdir )
    Script.load()

    Structure = tsl.script.Structure.Structure( name, outdir )
    Structure.load()

    Presences = tsl.script.Presences.Presences( name, outdir )
    Presences.load()

    presence_sn = Presences.presence_sn
    structure = Structure.structure
    text = Script.script_lines

    print "%s has %s scenes" % ( name, len( presence_sn.keys() ) )

    for c in coefs:
        partitions = get_partitions( presence_sn, c )
        print "For coefficient %s there were %s dramatic units in %s" % ( c, len( partitions ), name )
        
        print_script( structure, text, name, partitions, c )


