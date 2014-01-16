import re

import tsl.script.Script

class Interactions( tsl.script.Script.Script ):
    '''i = Interactions( 'Name of Movie', outdir='/tmp/movie-stuff' ) # Outdir defaults to .
    i.interactions = foo
    i.interaction_ns = bar
    i.interaction_sn = baz

    i.save( outdir='/tmp/movie-stuff', pretty=True ) 
         # Outdir defaults to the Outdir set in the constructor, or . if none was set
         # Pretty controls whether the output JSON is human readable
         # Creates files in outdir called 'name_of_movie_script_interaction[s|_ns|_sn].json'

    i.load( outdir='/tmp/movie-stuff', loadfiles={ 'interactions' : '../p.json', 'interaction_ns' : 'ns.json', 'interaction_sn' : '/tmp/sn.json' } )
         # Outdir and filenames have same defaults as the save method
    '''


    def __init__( self, script, outdir=None ):
        self.script = script
        self.script_fname = re.sub( r'\s+', '_', script.lower() )

        self.interactions = []
        self.interaction_sn = {}
        self.interaction_ns = {}

        self.outdir = outdir
        self.outputs = [ 'interactions', 'interaction_sn', 'interaction_ns' ]

