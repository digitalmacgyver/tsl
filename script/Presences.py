import re

import tsl.script.Script

class Presences( tsl.script.Script.Script ):
    '''p = Presences( 'Name of Movie', outdir='/tmp/movie-stuff' ) # Outdir defaults to .
    p.presences = foo
    p.presence_ns = bar
    p.presence_sn = baz

    p.save( outdir='/tmp/movie-stuff', pretty=True ) 
         # Outdir defaults to the Outdir set in the constructor, or . if none was set
         # Pretty controls whether the output JSON is human readable
         # Creates files in outdir called 'name_of_movie_script_presence[s|_ns|_sn].json'

    p.load( outdir='/tmp/movie-stuff', loadfiles={ 'presences' : '../p.json', 'presence_ns' : 'ns.json', 'presence_sn' : '/tmp/sn.json' } )
         # Outdir and filenames have same defaults as the save method
    '''

    def __init__( self, script, outdir=None ):
        self.script = script
        self.script_fname = re.sub( r'\s+', '_', script.lower() )

        self.presences = []
        self.presence_sn = {}
        self.presence_ns = {}

        self.outdir = outdir

        self.outputs = [ 'presences', 'presence_sn', 'presence_ns' ]

            
