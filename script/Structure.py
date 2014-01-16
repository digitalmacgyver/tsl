import re

import tsl.script.Script

class Structure( tsl.script.Script.Script ):
    '''s = Structure( 'Name of Movie', outdir='/tmp/movie-stuff' ) # Outdir defaults to .
    s.structure = foo

    s.save( outdir='/tmp/movie-stuff', pretty=True ) 
         # Outdir defaults to the Outdir set in the constructor, or . if none was set
         # Pretty controls whether the output JSON is human readable
         # Creates files in outdir called 'name_of_movie_script_structure.json'

    s.load( outdir='/tmp/movie-stuff', loadfiles={ 'structures' : '../s.json' } )         # Outdir and filenames have same defaults as the save method
    '''

    def __init__( self, script, outdir=None ):
        self.script = script
        self.script_fname = re.sub( r'\s+', '_', script.lower() )

        self.structure = []

        self.outdir = outdir

        self.outputs = [ 'structure' ]

            
