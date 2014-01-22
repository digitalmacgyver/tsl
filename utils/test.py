#!/usr/bin/python

import json
import nltk
import pdb
import re
import sys

scripts = [
    ( 'The Big Lebowski', '../example-scripts/the_big_lebowski.txt' ),
    ( 'Chinatown', '../example-scripts/chinatown.txt' ),
    ( 'Dune', '../example-scripts/dune.txt' ),
    ( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    ( 'The Matrix', '../example-scripts/the_matrix.txt' ),
    ]


def process( script ):
    #pdb.set_trace()
    f = open( script )
    raw = f.read()
    raw_tokens = nltk.wordpunct_tokenize( raw )
    tok = [ t for t in raw_tokens if re.search( r'\w', t ) ]
    text = nltk.Text( tok )
    # text.collocations()
    #words = [ w.lower() for w in text ]
    words = text
    # vocab = sorted( set( words ) ) 
    # 
    # Stemming
    # porter = nltk.PorterStemmer()
    # [ porter.stem( w ) for w in words ]
    tagged_words = nltk.pos_tag( words )
    porter = nltk.PorterStemmer()
    # [ porter.stem( w ) for w in words ]
    stemmed_tags = [ ( porter.stem( t[0].lower() ), t[1] ) for t in tagged_words ]
    fd = nltk.FreqDist( stemmed_tags )
    verbs = [ word for ( word, tag ) in fd if ( tag.startswith('V') and len( word ) > 3 )]
    adj = [ word for ( word, tag ) in fd if ( tag.startswith('JJ') and len( word ) > 1 ) ]
    adv = [ word for ( word, tag ) in fd if ( tag.startswith('RB') and len( word ) > 1 ) ]
    nouns = [ word for ( word, tag ) in fd if ( tag in ('NN', 'NNS' ) and len( word ) > 1 ) ]
    return ( verbs, adj, adv, nouns )

for script in scripts:
    print script[0]
    ( verbs, adj, adv, nouns ) = process( script[1] )
    print "Adjectives      : ", adj[:10]
    print "Non-proper nouns: ", nouns[:10]
    print "Verbs           : ", verbs[:10]
