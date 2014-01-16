import json
import os
import re

class Script( object ):
    '''p = Scripts( 'Name of Movie', outdir='/tmp/movie-stuff' ) # Outdir defaults to .
    p.script_lines = foo

    p.save( outdir='/tmp/movie-stuff', pretty=True ) 
         # Outdir defaults to .
         # Pretty controls whether the output JSON is human readable
         # Creates files in outdir called 'name_of_movie_script_lines.json'

    p.load( outdir='/tmp/movie-stuff', loadfiles={ 'script_lines' : '../sl.json' } )
         # Outdir and filenames have same defaults as the save method
    '''

    def __init__( self, script, outdir=None ):
        self.script = script
        self.script_fname = re.sub( r'\s+', '_', script.lower() )

        self.script_lines = []

        self.outdir = outdir

        self.outputs = [ 'script_lines' ]


    def save( self, outdir=None, pretty=True ):
        if outdir is None:
            if self.outdir is None:
                outdir = '.'
            else:
                outdir = self.outdir
                
        if not os.path.isdir( outdir ):
            os.makedirs( outdir )

        for output in self.outputs:
            outfile = "%s/%s_%s.json" % ( outdir, self.script_fname, output )
            f = open( outfile, 'w' )
            if pretty:
                json.dump( self.__getattribute__( output ), f, sort_keys=True, indent=4 )
            else:
                json.dump( self.__getattribute__( output ), f )

            f.close()

    def load( self, loaddir=None, loadfiles=None ):
        if loaddir is None:
            if self.outdir is None:
                loaddir = '.'
            else:
                loaddir = self.outdir

        if loadfiles is None:
            loadfiles = {}

        input_files = self.outputs
        
        for input_file in input_files:
            if input_file not in loadfiles:
                loadfiles[input_file] = "%s/%s_%s.json" % ( loaddir, self.script_fname, input_file )

            f = open( loadfiles[input_file], 'r' )
            setattr( self, input_file, json.load( f ) ) 
            f.close()
            
