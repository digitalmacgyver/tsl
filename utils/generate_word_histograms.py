#!/usr/bin/python

import json
import re
import sys

import tsl.script.Presences
import tsl.script.Interactions
import tsl.script.Script
import tsl.script.Structure

from tsl.script.parse.const import CHARACTER, DISCUSS, LOCATION, SETTING
from tsl.script.reports.reports import top_presences, top_interactions, get_presence_csv, get_interaction_csv


scripts = [
    ( 'The Big Lebowski', '../example-scripts/the_big_lebowski.txt' ),
    ( 'Chinatown', '../example-scripts/chinatown.txt' ),
    ( 'Dune', '../example-scripts/dune.txt' ),
    ( 'Ghostbusters', '../example-scripts/ghostbusters.txt' ),
    ( 'The Matrix', '../example-scripts/the_matrix.txt' ),
    ]

# We want:
# f = open( 'document.txt' )
# raw = f.read()
# Collocations from a script
# tokens = nltk.word_tokenize( raw ) | ntlk.wordpunct_tokenize( raw )
# lctok = [ t.lower() for t in tokens ]
# text = nltk.Text( tokens )
# text.collocations()
#
# words = [ w.lower() for w in text ]
# vocab = sorted( set( words ) ) 
# 
# Stemming
# porter = nltk.PorterStemmer()
# [ porter.stem( t ) for t in tokens ]
#
# Part of Speech
# nltp.pos_tag( text ) 
# Yeilds: [ ( 'And' ,'CC' ), ( 'now', 'RB' ), ...
# DO POS TAGGING BEFORE STEMMING.
#1. 	CC 	Coordinating conjunction
#2. 	CD 	Cardinal number
#3. 	DT 	Determiner
#4. 	EX 	Existential there
#5. 	FW 	Foreign word
#6. 	IN 	Preposition or subordinating conjunction
#7. 	JJ 	Adjective
#8. 	JJR 	Adjective, comparative
#9. 	JJS 	Adjective, superlative
#10. 	LS 	List item marker
#11. 	MD 	Modal
#12. 	NN 	Noun, singular or mass
#13. 	NNS 	Noun, plural
#14. 	NNP 	Proper noun, singular
#15. 	NNPS 	Proper noun, plural
#16. 	PDT 	Predeterminer
#17. 	POS 	Possessive ending
#18. 	PRP 	Personal pronoun
#19. 	PRP$ 	Possessive pronoun
#20. 	RB 	Adverb
#21. 	RBR 	Adverb, comparative
#22. 	RBS 	Adverb, superlative
#23. 	RP 	Particle
#24. 	SYM 	Symbol
#25. 	TO 	to
#26. 	UH 	Interjection
#27. 	VB 	Verb, base form
#28. 	VBD 	Verb, past tense
#29. 	VBG 	Verb, gerund or present participle
#30. 	VBN 	Verb, past participle
#31. 	VBP 	Verb, non-3rd person singular present
#32. 	VBZ 	Verb, 3rd person singular present
#33. 	WDT 	Wh-determiner
#34. 	WP 	Wh-pronoun
#35. 	WP$ 	Possessive wh-pronoun
#36. 	WRB 	Wh-adverb 

# We care about:
# adjectives (new, good, high) - starts with JJ
# adverbs (really, already, still) - starts with RB
# nouns and proper nouns - Starts with NN

# fd = nltk.FreqDist( tagged_words )
# verbs = [ word for ( word, tag ) in fd if tag.startswith('V') ]
# adj = [ word for ( word, tag ) in fd if tag.startswith('JJ') ]
# adv = [ word for ( word, tag ) in fd if tag.startswith('RB') ]
# nouns = [ word for ( word, tag ) in fd if tag.startswith('NN') ]

def process( script ):
    pdb.set_trace()
    f = open( script )
    raw = f.read()
    raw_tokens = nltk.wordpunct_tokenize( raw )
    tok = [ t for t in raw_tokens if re.search( r'\w', t ) ]
    text = nltk.Text( tok )
    # text.collocations()
    words = [ w.lower() for w in text ]
    # vocab = sorted( set( words ) ) 
    # 
    # Stemming
    porter = nltk.PorterStemmer()
    tagged_words = [ porter.stem( w ) for w in words ]
    fd = nltk.FreqDist( tagged_words )
    verbs = [ word for ( word, tag ) in fd if tag.startswith('V') ]
    adj = [ word for ( word, tag ) in fd if tag.startswith('JJ') ]
    adv = [ word for ( word, tag ) in fd if tag.startswith('RB') ]
    nouns = [ word for ( word, tag ) in fd if tag.startswith('NN') ]
    return ( verbs, adj, adv, nouns )


def process_script( script ):
    #import pdb
    #pdb.set_trace()

    name = script[0]
    script_file = script[1]

    print "Working on:", name

    outdir = '../example-scripts/parsed/' + re.sub( r'\s+', '_', name.lower() )

    Script = tsl.script.Script.Script( name, outdir )
    Script.load()

    Structure = tsl.script.Structure.Structure( name, outdir )
    Structure.load()

    Presences = tsl.script.Presences.Presences( name, outdir )
    Presences.load()

    Interactions = tsl.script.Interactions.Interactions( name, outdir )
    Interactions.load()

    presence_png = tsl.script.reports.reports.presence_plot( Script, map( lambda x: x[0], top_presences( Presences, top_n=8, noun_types=[CHARACTER] ) ), "Top 8 Character Presence in "+name )
    f = open( outdir+'/character_presence.png', 'w' )
    f.write( presence_png.getvalue() )
    presence_png.close()
    presence_png = None
    f.close()

    f = open( outdir+'/presence.csv', 'w' )
    f.write( get_presence_csv( Presences ) )
    f.close()

    f = open( outdir+'/interaction.csv', 'w' )
    f.write( get_interaction_csv( Presences, Interactions ) )
    f.close()

    output_top_presences( top_presences( Presences, top_n=5, noun_types=[CHARACTER] ), outdir+'/top5_characters.csv' )
    output_top_presences( top_presences( Presences, top_n=5, presence_types=[DISCUSS] ), outdir+'/top5_speakers.csv' )
    output_top_presences( top_presences( Presences, top_n=5, noun_types=[LOCATION] ), outdir+'/top5_locations.csv' )

    output_top_interactions( top_interactions( Presences, Interactions, top_n=5, interaction_types=[SETTING] ), outdir+'/top5_hangouts.csv' )
    output_top_interactions( top_interactions( Presences, Interactions, top_n=5, noun_types=[ ( CHARACTER, CHARACTER ) ] ), outdir+'/top5_bffs.csv' )
    output_top_interactions( top_interactions( Presences, Interactions, top_n=5, interaction_types=[DISCUSS] ), outdir+'/top5_speakers.csv' )

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
