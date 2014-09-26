#!/usr/bin/env python

import hcluster
import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import os
import re
import shutil
import sys

import pprint
pp = pprint.PrettyPrinter( indent=4 )

# Dimensions
#
# Don't change the order of things here unless you also change the
# dist_funcs key lookups in register_dist_funcs
dimensions = [
    'named_characters',
    'distinct_locations',
    'location_changes',
    'percent_dialog',
    'distinct_words',
    'dramatic_units',
    'adj-adv_noun-verb_ratio',
    'supporting_characters',
    'hearing',
    'character_x_speakers',
    'scenes_percentage_for_characters',
    'percent_dialog_by_character',
    'dialog_words_score',
    'scene_dialog_score',
]

proj_dimensions = [
    'eccentricity',
    'density'
]

zero = {
    'named_characters' : 0,
    'distinct_locations' : 0,
    'location_changes' : 0,
    'percent_dialog' : 0,
    'distinct_words' : 0,
    'dramatic_units' : 0,
    'adj-adv_noun-verb_ratio' : 0,
    'supporting_characters' : 0,
    'hearing' : 0,
    'character_x_speakers' :  [ { 'speakers' : 0 }, { 'speakers' : 0 }, { 'speakers' : 0 }, { 'speakers' : 0 }, { 'speakers' : 0 } ],
    'scenes_percentage_for_characters' : [ { 'percentage_of_scenes' : 0 }, { 'percentage_of_scenes' : 0 }, { 'percentage_of_scenes' : 0 }, { 'percentage_of_scenes' : 0 }, { 'percentage_of_scenes' : 0 } ],
    'percent_dialog_by_character' : [ { 'percent_dialog' : 0 }, { 'percent_dialog' : 0 }, { 'percent_dialog' : 0 }, { 'percent_dialog' : 0 }, { 'percent_dialog' : 0 } ],
    'scene_dialog_score' : 0,
    'dialog_words_score' : [0, 0],
    'eccentricity' : 0,
    'density' : 0
    }


def r_dist( movie, dim ):
    '''Measures the distacnce for movie along dimension from zero.'''
    if dim in dist_funcs:
        new_value = dist_funcs[dim]( zero[dim], movie[dim] )
    else:
        new_value = default_dist( zero[dim], movie[dim] )
    return new_value

def dispersion_plot( movies, dimensions, title="", outdir="", filename="" ):
    # DEBUG - leave this alone for now - it was modified to make N
    # graphs one for each movie.

    return 

    movie_by_ecc = sorted( [ movies[x] for x in movies.keys() ], key=lambda k : k['eccentricity'] )

    movie_y_data = movie_by_ecc

    for i in [ 0 ] + range( len( movie_y_data ) ):

        xc = [ range( len( dimensions ) ) ] * len( movie_y_data )
        yc = [ [ r_dist( x, dimensions[d] ) for d in range( len( dimensions ) ) ] for x in movie_y_data ]

        marker = [ 'b_' ] * len( movie_y_data )
        marker[i] = 'ro'
        
        for j in range( len( movie_y_data ) ):
            plt.plot( xc[j], yc[j], marker[j], scalex=.1)
        plt.xticks( numpy.arange( len( dimensions ) ), dimensions, rotation=90 )
        plt.ylim( -0.2, 1.2 )
        plt.xlim( -1, len( dimensions ) )
        plt.title( title + "%s" % ( movie_y_data[i]['title'] ) )
        plt.tight_layout()

        # DEBUG - I broke this -it's half working.
        #movie_title = re.sub( r'\s+', '_', name.lower() )

        plt.savefig( "%s/%s" % ( outdir, filename ), format='png')

        plt.clf()

def projection_plot( movies, dimensions, title="Density versus Eccentricity", outdir="" ):
    if len( dimensions ) < 2:
        return

    # Array of normalized movie values:
    movie_data = [ movies[x] for x in movies.keys() ]
    
    yc = [ r_dist( y, dimensions[0] ) for y in movie_data ]
    xc = [ r_dist( x, dimensions[1] ) for x in movie_data ]

    filename = re.sub( r'\s+', '_', title )
    if filename[-4:] != '.png':
        filename += '.png'

    plt.plot( xc, yc, "bo", scalex=.1 )
    plt.xlim( -0.1, 1.1 )
    plt.ylim( -0.1, 1.1 )
    plt.title( title )
    plt.tight_layout()

    proj_title = re.sub( r'\s+', '_', title.lower() )

    plt.savefig( "%s/projection-%s.png" % ( outdir, proj_title ), format='png' )

    plt.clf()

def get_movies( movies_dir, partition="all" ):
    '''Returns a hash keyed on movie title whose body is the Python
    data structure made up of the _metrics.json for this film in the
    movies_dir.

    Parition is one of 'released', 'blacklist', or anything else.  If
    it is released only certain films will be operated on, if it is
    blacklist the inverse of that set is operated on, if it is
    something else all films are operated on.
    '''
    movies = {}
    for dirpath, dirnames, filenames in os.walk( movies_dir):
        for directory in dirnames:
            metrics_files = [ x for x in os.listdir( os.path.join( dirpath, directory ) ) if x.endswith( '_metrics.json' ) ]
            if len( metrics_files ) == 0:
                print "Skipping %s/%s" % ( dirpath, directory )
                continue
                
            metrics = json.load( open( os.path.join( dirpath, directory, metrics_files[0] ) ) )
            
            #if metrics['title'] not in ['Ghostbusters', 'Dune', 'Starwars', 'Vertigo', 'All is Lost']:
            #if metrics['title'] not in ['Ghostbusters', 'Dune', 'Starwars' ]:
            #    continue
            
            released = [
                'Chinatown', 
                'Dune', 
                'Ghostbusters', 
                'The Matrix', 
                'Good Will Hunting', 
                'The Book of Eli', 
                'Starwars', 
                'Alien', 
                'Vertigo', 
                'Terminator 2', 
                'Ratatouille', 
                'Analyze That', 
                'Batman Begins', 
                'Death to Smoochy', 
                'Get Carter', 
                'Gothika', 
                'Groundhogs Day', 
                'Red Planet', 
                'Smurfs', 
                'Sweet November', 
                'Taking Lives', 
                'Thirteen Ghosts', 
                '42', 
                'Frozen', 
                'Fruitvale Station', 
                'All is Lost', 
                'Amour', 
                'Argo', 
                'August Osage County', 
                'Celest and Jesse Forever', 
                'Chronicle', 
                'Dallas Buyers Club', 
                'Despicable Me 2', 
                'The Wolf of Wall Street', 
                'Prince of Persia', 
                'Oz the Great and Powerful', 
                'Nebraska', 
                'Monsters University', 
                'Magic Mike', 
                'Lone Survivor', 
                'Kill Your Darlings', 
                'Kick Ass 2', 
                'The Great Gatsby', 
                'The Invisible Woman', 
                'The Past', 
                'Twilight', 
                'Wadjda', 
                'Woman in Black', 
                'Prisoners', 
                'Real Steel', 
                'Rush', 
                'Rust and Bone', 
                'Skyfall', 
                'Smashed', 
                'Snow White and the Huntsman', 
                'The Croods', 
                'Beautiful Creatures',
                'The Killing Floor'
                ]
            
            if partition == "released" and metrics['title'] not in released:
                continue
            elif partition == "blacklist" and metrics['title'] in released:
                continue

            movies[metrics['title']] = metrics

    return movies

def normalize_movies( movies, dist_funcs, dimensions, stretch = False ):
    '''Takes in the movies data structure, and normalizes their
    coordinates to fall in the 0-1 range. If the coordinate in
    question is itself a vector, we normalize them such that the
    longest is a vector of cartesian length 1.

    If stretch is set to true, the range is further stretched so the
    values occupy the range from 0-1 by subtracting min_value from
    each value first.
    '''
    # Calculate the maximum and minimum value.
    max_dimensions = {}
    min_dimensions = {}

    for k in sorted( movies.keys() ):
        movie = movies[k]
        for dim in dimensions:
            coordinate = movie[dim]

            value = r_dist( movie , dim )

            if value > max_dimensions.get( dim, -1 ):
                max_dimensions[dim] = value
            if value < min_dimensions.get( dim, 999999999 ):
                min_dimensions[dim] = value


    # Reduce things beyond perfect scaling to prevent floating point
    # rounding errors from nailing us - we want points on [0, 1) not
    # [0, 1]
    fudge_factor = 0.99

    if stretch:
        for k in sorted( movies.keys() ):
            movie = movies[k]
            for dim in dimensions:
                value = movie[dim]
                
                new_value = r_dist( movie, dim )
            
                if dim == 'character_x_speakers':
                    if new_value != 0:
                        movie[dim] = [ { 'speakers' : x['speakers'] * ( new_value - min_dimensions[dim] ) / new_value } for x in movie[dim] ]
                    else:
                        movie[dim] = zero[dim]
                elif dim == 'scenes_percentage_for_characters':
                    if new_value != 0:
                        movie[dim] = [ { 'percentage_of_scenes' : x['percentage_of_scenes'] * ( new_value - min_dimensions[dim] ) / new_value } for x in movie[dim] ]
                    else:
                        movie[dim] = zero[dim]
                elif dim == 'percent_dialog_by_character':
                    if new_value != 0:
                        movie[dim] = [ { 'percent_dialog' : x['percent_dialog'] * ( new_value - min_dimensions[dim] ) / new_value } for x in movie[dim] ]
                    else:
                        movie[dim] = zero[dim]
                elif dim == 'dialog_words_score':
                    if new_value != 0:
                        movie[dim] = [ x * ( new_value - min_dimensions[dim] ) / new_value for x in movie[dim] ]
                    else:
                        movie[dim] = zero[dim]
                else:
                    movie[dim] = value - min_dimensions[dim]

    # Normalize the values.
    for k in sorted( movies.keys() ):
        movie = movies[k]
        for dim in dimensions:

            value = movie[dim]

            if stretch:
                scale = max_dimensions[dim] - min_dimensions[dim]
            else:
                scale = max_dimensions[dim]

            if dim == 'character_x_speakers':
                movie[dim] = [ { 'speakers' : fudge_factor * float( x['speakers'] ) / scale } for x in movie[dim] ]
            elif dim == 'scenes_percentage_for_characters':
                movie[dim] = [ { 'percentage_of_scenes' : fudge_factor * float( x['percentage_of_scenes'] ) / scale } for x in movie[dim] ]
            elif dim == 'percent_dialog_by_character':
                movie[dim] = [ { 'percent_dialog' : fudge_factor * float( x['percent_dialog'] ) / scale } for x in movie[dim] ]
            elif dim == 'dialog_words_score':
                movie[dim] = [ fudge_factor * float( x ) / scale for x in movie[dim] ]
            else:
                movie[dim] = fudge_factor * float( value ) / scale

            new_value = r_dist( movie, dim )
            
            if new_value > 1:
                print "ERROR - value over 1 post normalization."
                print "movie, dim, value, new_value, max:", movie, dim, value, new_value, max_dimensions[dim]
                sys.exit( 0 )
            elif new_value == 1:
                print "WARNING - value of 1 post normalization."

def default_dist( a, b ):
    return abs( a-b )

def register_dist_funcs( dist_funcs ):
    def five_vect( a, b, lookup ):
        result_dist = 0
        for i in range( 0, 5 ):
            a_val = None
            if i >= len( a ):
                a_val = 0
            else:
                if i == 0:
                    a_val = 3*a[i][lookup]
                else:
                    a_val = a[i][lookup]
            b_val = None
            if i >= len( b ):
                b_val = 0
            else:
                if i == 0:
                    b_val = 3*b[i][lookup]
                else:
                    b_val = b[i][lookup]
            result_dist += default_dist( a_val, b_val )**2
        return result_dist**0.5

    def character_x_speakers( a, b ):
        return five_vect( a, b, 'speakers' )
    dist_funcs[ 'character_x_speakers' ] = character_x_speakers

    def scenes_percentage_for_characters( a, b ):
        return five_vect( a, b, 'percentage_of_scenes' )
    dist_funcs[ 'scenes_percentage_for_characters' ] = scenes_percentage_for_characters

    def percent_dialog_by_character( a, b ):
        return five_vect( a, b, 'percent_dialog' )
    dist_funcs[ 'percent_dialog_by_character' ] = percent_dialog_by_character

    def dialog_words_score( a, b ):
        return ( ( a[0] - b[0] )**2 + ( a[1] - b[1] )**2 )**0.5
    dist_funcs[ 'dialog_words_score' ] = dialog_words_score

def cartesian_distance( dists ):
    '''Takes in an array of distances between coordinates, and
    aggregates them into a single distance function.  Here we use
    Cartesian distance.'''
    total_dist = 0
    for dist in dists:
        total_dist += dist**2
    return total_dist**0.5

def compute_distances( movies, dist_funcs, distance_func, dimensions ):
    '''Returns a hash of hash.  The keys are every pair of movies, and
    the value is distance between them.'''
    distances = {}
    for k1 in sorted( movies.keys() ):
        for k2 in sorted( movies.keys() ):
            m1 = movies[k1]
            m2 = movies[k2]
            dists = []
            for dim in dimensions:
                if dim in dist_funcs:
                    dists.append( dist_funcs[dim]( m1[dim], m2[dim] ) )
                else:
                    dists.append( default_dist( m1[dim], m2[dim] ) )
            distance = distance_func( dists )
            if k1 in distances:
                distances[k1][k2] = distance
            else:
                distances[k1] = { k2 : distance }
    return distances

def eccentricity( distances ):
    '''Returns a hash of movie, eccentricity.'''
    result = {}
    denominator = len( distances.keys() ) - 1
    for k1 in sorted( distances.keys() ):
        numerator = 0
        for k2, distance in distances[k1].items():
            if k1 == k2:
                continue
            numerator += distance
        result[k1] = numerator / denominator
    return result

def density( distances ):
    '''Returns a hash of movie, density.'''
    result = {}
    for k1 in sorted( distances.keys() ):
        numerator = 0
        for k2, distance in distances[k1].items():
            if k1 == k2:
                continue
            try:
                numerator += 1 / math.e**( distance**2 )
            except:
                # If we have an overflow don't worry about it, just
                # add nothing.
                raise Exception( "WHOOPS!" )
                pass
        result[k1] = numerator
    return result

def compute_projection( distances, projection_func ):
    return projection_func( distances )

def get_clusters( movies_input, distances, epsilon, method=None ):
    '''Given a hash of movie keys, the distances data structure, and
    epsilon threshold distance, returns an array of hashes of movie
    keys where each hash is a subset of the input movies containing
    the points which are within a transitive closure of epsilon of
    one another.'''

    if method is None:
        # Don't change the input value.
        movies = {}
        for movie in movies_input.keys():
            movies[movie] = True
    
        clusters = []

        while len( movies ):
            current_cluster = {}

            cluster_changed = True
            while cluster_changed:
                cluster_changed = False
                movie_keys = movies.keys()
                for movie in movie_keys:
                    if len( current_cluster ) == 0:
                        cluster_changed = True
                        current_cluster[movie] = True
                        del movies[movie]
                    else:
                        for cluster_movie in current_cluster.keys():
                            if distances[cluster_movie][movie] <= epsilon:
                                cluster_changed = True
                                current_cluster[movie] = True
                                if movie in movies:
                                    del movies[movie]
            clusters.append( current_cluster )
    else:
        print "ERROR - nondefault clustermethods not implemented yet."
        sys.exit( 0 )
        # method is the hcluster algorithm to use.
        # Transform our distances into a square distance matrix.
        '''
        square_dist = []
        print movies_input.keys()
        for a in sorted( movies_input.keys() ):
            current = []
            for b in sorted( movies_input.keys() ):
                current.append( distances[a][b] )
            square_dist.append( current )
        condensed_dist = hcluster.squareform( square_dist )
        clusters = method( condensed_dist )
        print hcluster.dendrogram( clusters )
        plt.show()
        sys.exit( 0 )
        '''

    return clusters

def cluster_epsilon_finder( movies, distances ):
    '''Calculates epsilon via the following algorithm:

    1. Clusters are defined to be a non-empty set of points, initially
    we have one cluster per point.

    2. Distance between clusters is defined to be the minimum distance
    between any two points in either cluster.

    3. We iteratively aggregate the two clusters having the minimum
    distance until there is only one cluster, while recording the
    distances involved.

    4. We return the array of epsilons to the caller.
    '''
    # Handle pathological cases
    if not len( movies ):
        raise Exception("Expected at least one movie in cluster_epsilon_finder.")
    elif len( movies ) == 1:
        return [0]
    
    cluster_distances = []

    # Create the initial cluster.
    min_i = None
    min_j = None
    min_dist = None
    for i in movies.keys():
        for j in movies.keys():
            if i == j:
                continue
            else:
                if min_dist is None or distances[i][j] < min_dist:
                    min_dist = distances[i][j]
                    min_i = i
                    min_j = j
    clusters = [ { min_i : True, min_j : True } ]
    cluster_distances.append( distances[min_i][min_j] )

    for movie in movies.keys():
        if movie != min_i and movie != min_j:
            clusters.append( { movie : True } )
    
    # Process the rest of the points.
    while len( clusters ) > 1:
        min_dist = None
        min_i = None
        min_j = None
        
        for i_idx, cluster_i in enumerate( clusters ):
            for j_idx, cluster_j in enumerate( clusters ):
                if i_idx == j_idx:
                    continue
                else:
                    for i in cluster_i.keys():
                        for j in cluster_j.keys():
                            if min_dist is None or distances[i][j] < min_dist:
                                min_dist = distances[i][j]
                                min_i_idx = i_idx
                                min_j_idx = j_idx
                                min_cluster_i = cluster_i
                                min_cluster_j = cluster_j
                                min_i = i
                                min_j = j
        # There are a few cases:
        #
        # 1. min_cluster_i and j are in singleton clusters := make a
        # new cluster of the two of them.
        #
        # 2. min_cluster_i or j is a singleton, but the other is not
        # := add the singleton to the larger cluster.
        #
        # 3. Neither min_cluster_i or j is a singleton := merge the
        # two.
        cluster_distances.append( min_dist )
        if len( min_cluster_i.keys() ) == 1 and len( min_cluster_j.keys() ) == 1:
            min_cluster_i[min_j] = True
            clusters = clusters[:min_j_idx] + clusters[min_j_idx+1:]
        elif len( min_cluster_i.keys() ) == 1 and len( min_cluster_j.keys() ) > 1:
            min_cluster_j[min_i] = True
            clusters = clusters[:min_i_idx] + clusters[min_i_idx+1:]
        elif len( min_cluster_i.keys() ) > 1 and len( min_cluster_j.keys() ) == 1:
            min_cluster_i[min_j] = True
            clusters = clusters[:min_j_idx] + clusters[min_j_idx+1:]
        else:
            for j_point in min_cluster_j.keys():
                min_cluster_i[j_point] = True
            clusters = clusters[:min_j_idx] + clusters[min_j_idx+1:]

    return cluster_distances

def output_d3( outdir, filename, vertices, edges, cliques, header, html_filename ):
    # NOTE: "element" can't have .'s in it or it causes some browser
    # incompatibilities with id's that have .'s in them...
    element = re.sub( r'\.', '_', filename )
    f = open( outdir+filename, 'w' )
    json.dump( { "nodes" : vertices, "links" : edges, "cliques" : cliques }, f )
    f.close()
    f = open( outdir+html_filename, 'a' )
    html_body = '''
    <td>
      <p>
        %s
      </p>
      <p id="%s"></p>
      <script>
        script_graph( "%s", "#%s", 540, 540, false );
      </script>
    </td>
''' % ( header, element, filename, element )
    f.write( html_body )
    f.close()

def make_graphs( movies, distances, dimensions, proj_dimensions, epsilon, width, slide, shading_key=None ):
    # The substantial code complexity below is to allow this code to
    # work with any number of projected dimensions.
    
    # Slide is intended to be >= 0.5 - by design and conventional
    # practice we don't want more than 2 regions overlapping in one
    # dimension.

    # We consider intervals of width whose overlaps are made by
    # sliding the initial point slide percentage of width along.
    #
    # Start at 0 and proceed until we start an interval >=1
    start = 0
    step = width * slide

    movie_cell_map = {}
    cell_movie_map = {}

    #print "Step is:", step

    for movie in sorted( movies.keys() ):
        # We build an array of arrays, the i'th position of which is
        # the array of cells this movie is in for dimension i.
        movie_cells = []

        for dim in proj_dimensions:
            # Each movie is in 1 or 2 cells for each dimension.
            dimension_cells = []
            coordinate = movies[movie][dim]

            value = r_dist( movies[movie], dim )

            #print "Coordinate is:", coordinate
            #print "Value is:", value

            # Get the latest cell who starts at or before value.
            last_cell = int( math.floor( float( value ) / step ) )

            # See if this point is also in the prior cell.
            #
            # This code is written to accomodate the point being in
            # multiple prior cells, which could happen if more than
            # two cells overlap at a point.
            prior_offset = 1
            while ( step * ( last_cell - prior_offset ) + width > value ):
                dimension_cells.append( last_cell - prior_offset )
                prior_offset += 1

            dimension_cells.append( last_cell )
            movie_cells.append( dimension_cells )

        #print "MOVIE CELLS:"
        #pp.pprint( movie_cells )

        # Compose a list of cell keys where the cell key is of the
        # form: a_b_c_d..._z where a, b, c are cells this movie is
        # found in in each of the a..z dimentions
        next_cell_keys = [ [x] for x in movie_cells[0] ]
        for dim_cells in movie_cells[1:]:
            cell_keys = next_cell_keys
            next_cell_keys = []
            for dim_cell in dim_cells:
                for cell_key in cell_keys:
                    next_cell_keys.append( cell_key + [dim_cell] )

        cell_keys = next_cell_keys

        #print "CELL KEYS:"
        #pp.pprint( cell_keys )
                
        for cell_key in cell_keys:
            cell_string = '_'.join( [ str( x ) for x in cell_key ] )

            #print "cell_key:", cell_key
            #print "cell_string:", cell_string

            if movie in movie_cell_map:
                movie_cell_map[movie][cell_string] = True
            else:
                movie_cell_map[movie] = { cell_string : True }

            if cell_string in cell_movie_map:
                cell_movie_map[cell_string][movie] = True
            else:
                cell_movie_map[cell_string] = { movie : True }

        #pp.pprint( movie_cell_map )
            
    #pp.pprint( movie_cell_map )
    #pp.pprint( cell_movie_map )

    node_map = {}
    node_keys = {}
    movie_node_map = {}
    all_movie_clusters = []
    nodes = []
    edges = []
    cliques = []
    graph = nx.Graph()

    # If true only add nodes if they are not strict subsets of existing nodes.
    eliminate_subsets = False

    for cell_string in sorted( cell_movie_map ):
        cell_movie_keys = cell_movie_map[cell_string].keys()
                
        cell_movies = { key : value for ( key, value ) in movies.items() if key in cell_movie_keys }

        cell_movie_clusters = get_clusters( cell_movies, distances, epsilon )

        if eliminate_subsets:
            for cell_movie_cluster in cell_movie_clusters:
                cluster_movies = cell_movie_cluster.keys()

                add_node = True
                # Determine if this node is a subset of any existing
                # node.
                for existing_cluster in all_movie_clusters:
                    subset_of_current = True
                    for cluster_movie in cluster_movies:
                        if cluster_movie not in existing_cluster:
                            subset_of_current = False
                            break

                    if subset_of_current:
                        add_node = False
                        break

                # Delete any existing nodes that are subsets of this
                # node.
                def is_subset( a, b ):
                    for thing in a:
                        if thing not in b:
                            return False
                    return True

                if add_node:
                    all_movie_clusters = [ x for x in all_movie_clusters if not is_subset( x, cluster_movies ) ]

                    all_movie_clusters.append( cell_movie_cluster )
        else:
            all_movie_clusters += cell_movie_clusters

    for cell_movie_cluster in all_movie_clusters:
        cluster_movies = cell_movie_cluster.keys()
        label = ', '.join( sorted( cluster_movies ) )
                    
        add_node = False
        
        if label not in node_map:
            add_node = True

        if add_node:
            node_map[label] = len( nodes )
            node_keys[len( nodes )] = cell_movie_cluster

            shading = 0.5
            if shading_key:
                for cluster_movie in cluster_movies:
                    movie_shading = movies[cluster_movie].get( shading_key, 0.5 )
                    shading += movie_shading

                shading = float( shading ) / len( cluster_movies )
                shading = shading**2

            nodes.append( { "name" : label, 
                            "group" : 0,
                            "elements" : len( cluster_movies ),
                            "shading" : shading } )
            graph.add_node( len( nodes ) - 1 )
                    
            for cluster_movie in cluster_movies:
                if cluster_movie not in movie_node_map:
                    movie_node_map[cluster_movie] = { label : True }
                elif label not in movie_node_map[cluster_movie]:
                    for dest_label in movie_node_map[cluster_movie].keys():
                        edges.append( { "source" : node_map[label],
                                        "target" : node_map[dest_label],
                                        "value" : 1 } )
                        graph.add_edge( node_map[label], node_map[dest_label] )

                movie_node_map[cluster_movie][label] = True
                        
    # We don't care about trivial cliques.
    cliques = [ x for x in list( nx.find_cliques( graph ) ) if len( x ) > 2 ]
                                                
    # Things are only actually a clique for us if all connected nodes
    # also contain a particular movie.
    filtered_cliques = []
    for clique in cliques:
        #print "FOR CLIQUE:", clique
        #print [ set( node_keys[x].keys() ) for x in clique ]
        common_movies = reduce( set.intersection, [ set( node_keys[x].keys() ) for x in clique ] )
        if len( common_movies ) > 0:
            #print "COMMON MOVIES:", common_movies
            filtered_cliques.append( clique )
        else:
            #print "NO COMMON MOVIES", width, step, epsilon
            pass

    return ( nodes, edges, filtered_cliques )

def get_dimensions( measures, projection ):
    '''Return a list of permutations whereby each element of measures
    is removed and used as the second element of projection.'''

    result = []

    for i in range( len( measures ) ):
        dim_projections = [ projection, measures[i] ]
        result.append( ( measures[:i]+measures[i+1:], dim_projections, measures[i] ) )

    return result

def graph_html( outdir, mode="create", idx=0, label='', value=0, x='', y='' ):
    if mode == "create":
        f = open( outdir+"/graphs-%s-%s-%d.html" % ( x, y, idx ), 'w' )
    else:
        f = open( outdir+"/graphs-%s-%s-%d.html" % ( x, y, idx ), 'a' )

    if mode == "create":
        html_front = '''
<!DOCTYPE html>
<head>
<meta charset="utf-8">
<style>

.node {
  stroke: #fff;
  stroke-width: 1.5px;
}

.link {
  stroke: #999;
  stroke-opacity: .6;
}

</style>
'''
        html_front += '  <title>%s %s Projection, %s=%s</title>\n' % ( x.capitalize(), y.capitalize(), label.capitalize(), value )
        html_front += '''  <script src="http://d3js.org/d3.v3.min.js"></script>
  <script src="graph.js"></script>
</head>
<body>
  <h1>%s %s Projection, %s=%s</h1>
  <p><a href="index.html">Back to index</a></p>
  <table border="1px">
''' % ( x.capitalize(), y.capitalize(), label.capitalize(), value )
        f.write( html_front )
    elif mode == "start_row":
        html_body = "  <tr>\n"
        f.write( html_body )
    elif mode == "end_row":
        html_body = "  </tr>\n"
        f.write( html_body )
    elif mode == "end":
        html_back = '''
  </table>
</body>
'''
        f.write( html_back )

    f.close()

def index_html( outdir, mode="create", idx=0, label='', value=0, x='', y='' ):
    if mode == "create":
        f = open( outdir+"/index.html", 'w' )
        html_front = '''
<!DOCTYPE html>
<head>
<meta charset="utf-8" />
  <style>

.node {
  stroke: #fff;
  stroke-width: 1.5px;
}

.link {
  stroke: #999;
  stroke-opacity: .6;
}

  </style>
  <title>Story Topology</title>
</head>
<body>
  <ul>
'''
        f.write( html_front )
        f.close()
    elif mode == "append":
        f = open( outdir+"/index.html", 'a' )
        li = '<li><a href="graphs-%s-%s-%d.html">%s %s Projection, %s=%s</a></li>\n' % ( x, y, idx, x.capitalize(), y.capitalize(), label.capitalize(), value )
        f.write( li )
        f.close()
    elif mode == "end":
        f = open( outdir+"/index.html", 'a' )
        html_back = '''  </ul>
</body>
'''
        f.write( html_back )
        f.close()

if __name__=="__main__":

    # GOALS:
    #
    # 1. Print everything in a table arranged by eccentricity, width.
    #    * Pages are arranged by step.
    #    * Pages have URL back to parent.
    #
    # 2. Make N sets of charts, one for eccentricity + dim_i for measure I.
    #   * Set proj_title appropriately.

    # Read in movie JSON files.
    movies_dir = "../example-scripts/parsed"

    dist_funcs = {}
    register_dist_funcs( dist_funcs )

    partitions = [ 'released', 'blacklist', 'all' ]

    for partition in partitions:
        movies = get_movies( movies_dir, partition )

        #outdir = "/wintmp/movie/graph13/%s/" % ( partition )
        outdir = "/home/mhayward/movie/RackStatic/public/graph4/%s/" % ( partition )
        plotdir = '/wintmp/movie/plots/graph13/%s/' % ( partition )

        if not os.path.isdir( outdir ):
            os.makedirs( outdir )
        if not os.path.isdir( plotdir ):
            os.makedirs( plotdir ) 

        shutil.copy( os.path.split( __file__ )[0] + '/graph-new.js', "%s/graph.js" % ( outdir ) )

        normalize_movies( movies, dist_funcs, dimensions, stretch=True )

        index_html( outdir, "create" )

        for ( dim, proj, current_proj ) in get_dimensions( dimensions, 'eccentricity' ):
            # We could in principle have difference means of calculating our
            # distance.
            print "Working on dimensions: %s, %s" % ( 'eccentricity', current_proj )
            distance_func = cartesian_distance
            distances = compute_distances( movies, dist_funcs, distance_func, dim )
            #print "Distances:"
            #pp.pprint( distances )

            projection_func = eccentricity
            projection = compute_projection( distances, projection_func )

            for movie in projection.keys():
                movies[movie]['eccentricity'] = projection[movie]

            projection_func = density
            projection = compute_projection( distances, projection_func )

            for movie in projection.keys():
                movies[movie]['density'] = projection[movie]

            normalize_movies( movies, dist_funcs, proj, stretch=True )

            dispersion_plot( movies, proj + dim, outdir )
            projection_plot( movies, proj, title=" vs ".join( proj ), outdir=outdir )

            # We use the maximum difference between any two sequential
            # data points along one of our projection dimension to
            # help us pick our steps.
            #
            # By ensuring our widths are at least 2*max_proj_diffs we
            # ensure that each point in our space is in a tile with at
            # least one other point when step = 0.5.
            max_proj_diffs = -1
            for p in proj:
                ps = [ r_dist( movies[k], p ) for k in movies.keys() ]
                ps.sort()
                p_diffs = [ ps[x] - ps[x-1] for x in range( 1, len( ps ) ) ]
                p_diff = max( p_diffs )
                print "Maximum %s diff = %s" % ( p, p_diff )
                if p_diff > max_proj_diffs:
                    max_proj_diffs = p_diff

            epsilon_candidates = cluster_epsilon_finder( movies, distances )
            print "Cluster epsilon candidates", epsilon_candidates
            # NOTE: Just because the larges epsilon candidate is X
            # does not mean any two points will be linked together
            # with X in our graph - as we consider subsets of the
            # graph.  The ensure everything gets lumped together you'd
            # have to make epsilon the largest distance between any
            # two points.
            max_epsilon = epsilon_candidates[-1]
            epsilons = [ max_epsilon / 3 , max_epsilon / 2, 2 * max_epsilon / 3, max_epsilon ]
            epsilons.reverse()
            widths = [ max_proj_diffs / 16, max_proj_diffs / 8, max_proj_diffs / 4, max_proj_diffs, max_proj_diffs*2 ]
            widths.reverse()
            slides = [ .5, .75, .9 ]

            print "Steps   : %s" % ( slides )
            print "Widths  : %s" % ( widths )
            print "Epsilons: %s" % ( epsilons )

            for ( step_idx, slide ) in enumerate( slides ):
                index_html( outdir, "append", step_idx, 'slide', slide, 'eccentricity', current_proj )
                graph_html( outdir, "create", step_idx, 'slide', slide, 'eccentricity', current_proj )

                for epsilon in epsilons:
                    graph_html( outdir, "start_row", step_idx, 'slide', slide, 'eccentricity', current_proj )
                    for width in widths:
                        shading_key = None
                        if partition == 'released':
                            shading_key = 'rt'
                        nodes, edges, filtered_cliques = make_graphs( movies, distances, dim, proj, epsilon, width, slide, shading_key = shading_key )

                        filename = "_".join( [ 'eccentricity', current_proj, "%0.02f" % ( slide ), "%0.02f" % ( width ), "%0.02f" % ( epsilon ) ] ) + ".json"

                        print "Creating output for: %s-%s" % ( partition, filename )

                        output_d3( outdir, filename, nodes, edges, filtered_cliques, "Width: %s, Epsilon: %0.02f" % ( width, epsilon ), "graphs-%s-%s-%d.html" % ( 'eccentricity', current_proj, step_idx ) )
                        
                    graph_html( outdir, "end_row", step_idx, 'slide', slide, 'eccentricity', current_proj )
                
                graph_html( outdir, "end", step_idx, 'slide', slide, 'eccentricity', current_proj )
                
    index_html( outdir, "end" )

