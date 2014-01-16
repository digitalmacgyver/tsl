from illuminator.parse import CHARACTER, THING, LOCATION, SETTING, DISCUSS, MENTION, APPEAR, presences, presence_ns, presence_sn, interactions, interaction_ns, interaction_sn

#import matplotlib.pyplot as plt
import nltk.text
nltk.data.path.append( '/app/illuminator/nltk_data' )
from nltk.tokenize import word_tokenize
import re
import StringIO

def top_presences( top_n=0, min_appearances=1, noun_types=[], presence_types = [], scene_list=[] ):
    '''Return an array of name, noun_type, count tuples ordered in
    descending order of count.  

    Optionally this list is filtered to include only the top_n
    results, or results with at least min_appearances, or results
    matching types in noun_types or presence_types or are constrained
    to scenes in scene_list.'''

    result = []

    for name in presence_ns:
        appearances = 0

        ntype = presence_ns[name]['noun_type']
        if noun_types and not ntype in noun_types:
            continue
        
        scenes = []
        if scene_list:
            scenes = list( 
                set( 
                    scene_list 
                    ).intersection( 
                    set( 
                        presence_ns[name].keys() 
                        ) 
                    ) 
                )
        else:
            scenes = [ scene for scene in presence_ns[name].keys() if scene != 'noun_type' ]
        for scene in scenes:
            if presence_types:
                for appearance in presence_ns[name][scene]:
                    if appearance['presence_type'] in presence_types:
                        appearances += 1
            else:
                appearances += len( presence_ns[name][scene] )

        if appearances >= min_appearances:
            result.append( ( name, ntype, appearances ) )

    if top_n > 0:
        return sorted( result, key=lambda x: -x[2] )[:top_n]
    else:
        return sorted( result, key=lambda x: -x[2] )

def top_interactions( top_n=0, min_interactions=1, noun_types=[], presence_types=[], scene_list=[], interaction_types=[], names=None ):
    '''Return an array of name1, name2, count tuples in descending order of count.

    Return only the top_n interactors
    
    Return only interactions that occur at least min_interaction
    times.

    A list of noun type tuples where one noun must match one type of
    the tupe, and the other must match the other.  Note that there is
    no inherent order you could presence a ( CHARACTER, LOCATION )
    tuple and get back a relationship where a is a LOCATION and b is a
    CHARACTER - this is because all relationships are symmetric.

    A similar presence_type tuple list exists.

    A list of scenes can be provided, then only interactions in the
    aforementioned scenes are provided.

    Finally a list of interaction types can be provided to limit the
    types of interactions desired.'''


    result = []

    handled = {}

    if not names:
        names = sorted( interaction_ns.keys() )
    else:
        names.sort()

    for name1 in names:
        for name2 in sorted( interaction_ns[name1].keys() ):
            # We have all interactions symetrically, so a-b and b-a
            # will both appear.  For reporting we'll just show the a-b
            # interactions, not the b-a interactions.
            if name1 in handled and name2 in handled[name1]:
                continue    
            if name2 in handled:
                handled[name2][name1] = True
            else:
                handled[name2] = { name1 : True }

            interactions = 0
            
            if noun_types and not valid_noun_types( name1, name2, noun_types ):
                continue

            scenes = []
            if scene_list:
                scenes = list( set( scene_list ).intersection( set( interaction_ns[name1][name2].keys() ) ) )
            else:
                scenes = [ scene for scene in interaction_ns[name1][name2].keys() ]

            for scene in scenes:
                if interaction_types:
                    interactions += count_valid_interactions( [ i for i in interaction_ns[name1][name2][scene] if i['interaction_type'] in interaction_types ], presence_types )
                else:
                    interactions += count_valid_interactions( interaction_ns[name1][name2][scene], presence_types )

            if interactions >= min_interactions:
                result.append( ( name1, name2, interactions ) )

    if top_n > 0:
        return sorted( result, key=lambda x: -x[2] )[:top_n]
    else:
        return sorted( result, key=lambda x: -x[2] )

def count_valid_interactions( interactions, presence_types ):
    if presence_types:
        count = 0
        for interaction in interactions:
            for type1, type2 in presence_types:
                if (    ( interaction['a']['presence_type'] == type1 and interaction['b']['presence_type'] == type2 )
                     or ( interaction['a']['presence_type'] == type2 and interaction['b']['presence_type'] == type1 ) ):
                    count += 1
        return count
    else:
        return len( interactions )
        

def valid_noun_types( p1, p2, noun_types ):
    for type1, type2 in noun_types:
        if (    ( presence_ns[p1]['noun_type'] == type1 and presence_ns[p2]['noun_type'] == type2 )
             or ( presence_ns[p1]['noun_type'] == type2 and presence_ns[p2]['noun_type'] == type1 ) ):
            return True
        else:
            return False


def print_interaction_ns():
    x = ''
    for name1 in interaction_ns:
        for name2 in interaction_ns[name1]:
            for scene in interaction_ns[name1][name2]:
                for i in interaction_ns[name1][name2][scene]:
                    x += ' '.join([ i['a']['name'],i['b']['name'],i['interaction_type'],str( i['where']['line_no'] ) ])
                    x += "\n"
    return x

def get_presence_csv():
    '''Return a CSV like string with all presences, including noun,
    noun type, presence_type, and where information.'''
    ret = "name,nount_type,presence_type,scene_id,page_no,line_no\n"

    for p in presences:
        ret += ','.join( [ p['name'], presence_ns[p['name']]['noun_type'], p['presence_type'], p['where']['scene_id'], str( p['where']['page_no'] ), str( p['where']['line_no'] ) ] )
        ret += "\n"

    return ret

def get_interaction_csv():
    '''Return a CSV like string with all interactions including both
    nouns, types, presence_types, interaction type, and where
    information'''

    ret = "noun1,noun1_type,noun1_presence_type,noun2,noun2_type,noun2_presence_type,interaction_type,scene_id,page_no,line_no\n"

    for i in interactions:
        p1 = i['a']
        p2 = i['b']
        interaction_type = i['interaction_type']
        where = i['where']
        
        ret += ','.join( [ p1['name'], presence_ns[p1['name']]['noun_type'], p1['presence_type'], 
                           p2['name'], presence_ns[p2['name']]['noun_type'], p2['presence_type'], 
                           interaction_type, where['scene_id'], str( where['page_no'] ) , str( where['line_no'] ) ] )
        ret += "\n"

    return ret

def get_singletons():
    '''Return a list of nouns that only appear once.'''

    singletons = []

    for name in presence_ns:
        scenes = [ x for x in presence_ns[name].keys() if not x in ['noun_type'] ]
        if len( scenes ) == 1:
            if len( presence_ns[name][scenes[0]] ) == 1:
                singletons.append( name )

    return singletons

def presence_plot( script_lines, names, title ):
    '''Produce a plot of the names provided in the script based on
    their word offset.  Title must end in .png'''

    script = ''.join( map( lambda x: x['content'], script_lines ) ).upper()

    new_names = []

    for name in names:
        new_name = re.sub( r'\s+', '_', name )
        script = script.replace( name, new_name )
        new_names.append( new_name )

    return dispersion_plot( 
        word_tokenize( script ),
        new_names,
        title )

def dispersion_plot(text, words, title="Lexical Dispersion Plot"):
    """
    Generate a lexical dispersion plot.
    
    :param text: The source text
    :type text: list(str) or enum(str)
    :param words: The target words
    :type words: list of str
    :param ignore_case: flag to set if case should be ignored when searching text
    :type ignore_case: bool
    """
    text = list(text)
    words.reverse()
    
    words_to_comp = words
    text_to_comp = text

    points = [(x,y) for x in range(len(text_to_comp))
              for y in range(len(words_to_comp))
              if text_to_comp[x] == words_to_comp[y]]
    if points:
        x, y = zip(*points)
    else:
        x = y = ()

'''
    filename = re.sub( r'\s+', '_', title )
    if filename[-4:] != '.png':
        filename += '.png'

    plt.plot(x, y, "b|", scalex=.1)
    plt.yticks(range(len(words)), words, color="b")
    plt.ylim(-1, len(words))
    plt.title(title)
    plt.xlabel("Word Offset")
    output = StringIO.StringIO()

    plt.savefig(output, format='png')

    plt.clf()

    return output
'''
