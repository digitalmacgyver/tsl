#!/usr/bin/env python

import json
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy
import os
import re
import StringIO
import sys
import types

'''
Logic: 

4) Make:

DONE h[movie] = cluster_id
DONE h[cluster_id] = { movie1 : True, movie2: True, ... }

DONE 5) Choose overlap step: 0-1, 0+e-1+e, ...

6) Make data structure:

h[region_key][point] = True

for each movie:
   determine what cells it is in (between 1 and 2^14).

   Label each cell as [3, 4, 1, ... 19] of 14 elements, the i'th
   element is the position of the i'th dimension, and the value is the
   overlap region number.  This vector can be a label consolidated as
   n0-n1-n2...n14.

   h[movie][region_key] = True

   h[region_key][movie] = True

7) Then do:

for each movie:
   movie_cluster = h[movie]
   for each region the movie is in:
      for each other_movie with a point in that region:
          movie2_cluster = h[other_movie]
          
          graph.add_edge( movie_cluster, movie2_cluster )

'''

# Dimensions
#
# Don't change the order of things here unless you also change the
# dist_funcs key lookups in register_dist_funcs

#dimensions = [ 'main_character_interlocutor_count', 'percentage_of_scenes_with_main_character' ]

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
#    'scene_dialog_score',
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
    'dialog_words_score' : [0, 0]
    }


# Read in movie JSON files.
movies_dir = "../example-scripts/parsed"

outdir = "/wintmp/movie/graph6/"

def plot_dist( movie, dim ):
    if dim in dist_funcs:
        new_value = dist_funcs[dim]( zero[dim], movie[dim] )
    else:
        new_value = default_dist( zero[dim], movie[dim] )
    return new_value

def dispersion_plot( movies, title="Plot" ):

    # Array of normalized movie values:
    movie_y_data = [ movies[x] for x in movies.keys() ]
    
    xc = [ [x]*len( movies ) for x in range( len( dimensions ) ) ]
    yc = [ [ plot_dist( x, dimensions[d] ) for x in movie_y_data ] for d in range( len( dimensions ) ) ]

    filename = re.sub( r'\s+', '_', title )
    if filename[-4:] != '.png':
        filename += '.png'

    plt.plot(xc, yc, "b_", scalex=.1)
    #plt.yticks(range(len(yc)), yc, color="b")
    plt.xticks( numpy.arange( len( dimensions ) ), dimensions, rotation=90 )
    plt.ylim( -0.1, 1.1 )
    plt.xlim( -1, len( xc ) )
    plt.title(title)
    plt.tight_layout()

    #plt.xlabel("Word Offset")

    plt.savefig( "foo.png", format='png')

    plt.clf()

def get_movies( movies_dir ):
    '''Returns a hash keyed on movie title whose body is the Python
    data structure made up of the _metrics.json for this film in the
    movies_dir.'''
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

            movies[metrics['title']] = metrics

    return movies

def normalize_movies( movies, dist_funcs ):
    '''Takes in the movies data structure, and normalizes their
    coordinates to fall in the 0-1 range. If the coordinate in
    question is itself a vector, we normalize them such that the
    longest is a vector of cartesian length 1.
    '''

    # Calculate the maximum value.
    max_dimensions = {}

    for k in sorted( movies.keys() ):
        movie = movies[k]
        for dim in dimensions:
            coordinate = movie[dim]

            value = -1

            if dim in dist_funcs:
                value = dist_funcs[dim]( zero[dim], coordinate )
            else:
                value = default_dist( zero[dim], coordinate )
            
            if value > max_dimensions.get( dim, -1 ):
                max_dimensions[dim] = value

    # Reduce things beyond perfect scaling to prevent floating point
    # rounding errors from nailing us - we want points on [0, 1) not
    # [0, 1]
    fudge_factor = 0.99

    # Normalize the values.
    for k in sorted( movies.keys() ):
        movie = movies[k]
        for dim in dimensions:

            value = movie[dim]

            if dim == 'character_x_speakers':
                movie[dim] = [ { 'speakers' : fudge_factor * float( x['speakers'] ) / max_dimensions[dim] } for x in movie[dim] ]
            elif dim == 'scenes_percentage_for_characters':
                movie[dim] = [ { 'percentage_of_scenes' : fudge_factor * float( x['percentage_of_scenes'] ) / max_dimensions[dim] } for x in movie[dim] ]
            elif dim == 'percent_dialog_by_character':
                movie[dim] = [ { 'percent_dialog' : fudge_factor * float( x['percent_dialog'] ) / max_dimensions[dim] } for x in movie[dim] ]
            elif dim == 'dialog_words_score':
                movie[dim] = [ fudge_factor * float( x ) / max_dimensions[dim] for x in movie[dim] ]
            else:
                movie[dim] = fudge_factor * float( value ) / max_dimensions[dim]

            if dim in dist_funcs:
                new_value = dist_funcs[dim]( zero[dim], movie[dim] )
            else:
                new_value = default_dist( zero[dim], movie[dim] )
            
            if new_value > 1:
                print "ERROR - value over 1 post normalization."
                print "movie, dim, value, new_value, max:", movie, dim, value, new_value, max_dimensions[dim]
                sys.exit( 0 )
            elif new_value == 1:
                print "WARNING - value of 1 post normalization."

def default_dist( a, b ):
    return abs( a-b )

def register_dist_funcs( dist_funcs ):
    '''
    def log_dist( a, b ):
        return abs( math.log( a ) - math.log( b ) )

    dist_funcs[ dimensions[2] ] = log_dist
    dist_funcs[ dimensions[7] ] = log_dist
    '''
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
    dist_funcs[ dimensions[9] ] = character_x_speakers

    def scenes_percentage_for_characters( a, b ):
        return five_vect( a, b, 'percentage_of_scenes' )
    dist_funcs[ dimensions[10] ] = scenes_percentage_for_characters

    def percent_dialog_by_character( a, b ):
        return five_vect( a, b, 'percent_dialog' )
    dist_funcs[ dimensions[11] ] = percent_dialog_by_character

    def dialog_words_score( a, b ):
        return ( ( a[0] - b[0] )**2 + ( a[1] - b[1] )**2 )**0.5
    dist_funcs[ dimensions[12] ] = dialog_words_score

def cartesian_distance( dists ):
    '''Takes in an array of distances between coordinates, and
    aggregates them into a single distance function.  Here we use
    Cartesian distance.'''
    total_dist = 0
    for dist in dists:
        total_dist += dist**2
    return total_dist**0.5

def compute_distances( movies, dist_funcs, distance_func ):
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
    '''A hash of movie, eccentricity.'''
    result = {}
    denominator = len( distances.keys() )
    for k1 in sorted( distances.keys() ):
        numerator = 0
        for k2, distance in distances[k1].items():
            numerator += distance
        result[k1] = numerator / denominator
    return result

def density( distances ):
    '''A hash of movie, density.'''
    result = {}
    for k1 in sorted( distances.keys() ):
        numerator = 0
        for k2, distance in distances[k1].items():
            try:
                numerator += 1 / math.e**( distance**2 )
            except:
                # If we have an overflow don't worry about it, just
                # add nothing.
                pass
        result[k1] = numerator
    return result

def compute_projection( distances, projection_func ):
    return projection_func( distances )

def get_inverse_covering( projection, covering ):
    '''Given a covering, which is defined as an array of tuples, the
    elements a, b of which define the interval: [a, b], and a
    projection data structure, return:
    
    An array of hashes, the i'th element of which corresponds to the
    inverse image of the things in the projection for the i'th tuple.

    The format of these hashes is:
    { range: ( a, b ), movies: { 'Movie 1': True, 'Movie 2': True, ... } }'''

    inverse_covering = []

    for interval in covering:
        start = interval[0]
        end = interval[1]

        current_inverse = { 'range' : interval, 'movies' : {} }

        for movie, value in projection.items():
            if start <= value and value <= end:
                current_inverse['movies'][movie] = True
        
        inverse_covering.append( current_inverse )

    return inverse_covering
                
def get_clusters( movies_input, distances, epsilon ):
    '''Given a hash of movie keys, the distances data structure, and
    epsilon threshold distance, returns an array of hashes of movie
    keys where each hash is a subset of the input movies containing
    the points which are within a transitive closure of epsilon of
    one another.'''

    # Don't change the input value.
    movies = {}
    for movie in movies_input.keys():
        movies[movie] = True
    
    clusters = []

    #import pdb
    #pdb.set_trace()

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


        #for movie in movie_keys:
        #    if len( current_cluster ) == 0:
        #        current_cluster[movie] = True
        #        del movies[movie]
        #    else:
        #        for cluster_movie in current_cluster.keys():
        #            if distances[cluster_movie][movie] <= epsilon:
        #                current_cluster[movie] = True
        #                if movie in movies:
        #                    del movies[movie]
            
        clusters.append( current_cluster )

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

    4. We select the median distance recorded in step 3.
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

def make_covering( low, high, width, overlap ):
    step = float( width ) / overlap
    current = low
    covering = []
    while current < high:
        covering.append( ( current, current + width ) )
        current += step
    return covering

def output_d3( filename, vertices, edges, header ):
    f = open( outdir+filename+".json", 'w' )
    json.dump( { "nodes" : vertices, "links" : edges }, f )
    f.close()
    f = open( outdir+"graphs.html", 'a' )
    html_body = '''
<p>
%s
</p>
<script>
script_graph( "%s", width=768, height=432 );
</script>
''' % ( header, filename+".json" )
    f.write( html_body )
    f.close()

def make_graph( low, high, width, overlap, epsilon ):
    covering = make_covering( low, high, width, overlap )

    print "Covering is:", covering

    inverse_covering = get_inverse_covering( projection, covering )

    # Array of { "name":"Foo","group":cluster_idx }
    vertices = []
    # Array of { "source":idx of thing in vertices, "target":idx of thing in vertices", value:1 }
    edges = []

    graph = nx.Graph()

    label_to_vertex = {}

    for p_idx, partition in enumerate( inverse_covering ):
        partition_clusters = get_clusters( partition['movies'], distances, epsilon )
        print "Range from %s to %s had %s movies, which formed the following clusters:" % ( partition['range'][0], partition['range'][1], len( partition['movies'].keys() ) )

        for idx, cluster in enumerate( partition_clusters ):
            print "\tCluster %s" % idx
            label = 'Cover %s: ' % ( p_idx ) + ', '.join( sorted( cluster.keys() ) )

            #graph.add_node( label )
            #vertices.append( { "name" : label, "group" : p_idx } )
            #label_to_vertex[label] = len( vertices ) - 1

            #import pdb
            #pdb.set_trace()

            add_to_graph = True
            for node, data in graph.nodes( data=True ):
                same_as_existing = True
                for movie in cluster.keys():
                    if movie not in data:
                        same_as_existing = False
                for movie in data.keys():
                    if movie not in cluster:
                        same_as_existing = False
                if same_as_existing:
                    add_to_graph = False
                    print "Skipping cluster: %s as identical to %s" % ( label, node )
                    break


            if add_to_graph:
                graph.add_node( label )
                vertices.append( { "name" : label, "group" : p_idx, "elements" : len( cluster.keys() ), "shading" : float( partition['range'][0] ) / high } )
                label_to_vertex[label] = len( vertices ) - 1

            for movie in sorted( cluster.keys() ):
                if add_to_graph:
                    graph.node[label][movie] = True
                print "\t\t%s" % movie
                for node, data in graph.nodes( data=True ):
                    if movie in data and node != label and add_to_graph:
                        graph.add_edge( node, label )
                        edges.append( { "source" : label_to_vertex[node], "target" : label_to_vertex[label], "value" : 1 } )
 
                        
        #nx.write_dot( graph, 'file.dot' )           
        #positions = nx.graphviz_layout( graph, prog='neato' )
        #positions = nx.spring_layout( graph )
        #nx.draw( graph, pos=positions )
        #nx.draw_random( graph )
        #plt.show()

    #nx.draw_circle( graph )

    '''
    positions = nx.spring_layout( graph, scale=1024 )
    plt.figure( 1, figsize=(16,16) )
    nx.draw( graph, positions, font_size=8 )
    plt.show()
    #plt.figure( num=None, figsize=( 8, 8 ), facecolor='w', edgecolor='k' )
    #plt.savefig( "8x8_cover_width_%s_overlap_%s_epsilon_%0.02f.png" % ( width, overlap, epsilon ) )
    #plt.figure( num=None, figsize=( 16, 16 ) )
    #plt.savefig( "16x16_cover_width_%s_overlap_%s_epsilon_%0.02f.png" % ( width, overlap, epsilon ) )
    '''
    #import pdb
    #pdb.set_trace()
    plt.clf()
    positions = nx.spring_layout( graph, k=.1, iterations=100 )
    plt.figure( figsize=(16,9) )
    nx.draw( graph, pos=positions )
    filename = "cover_width_%s_overlap_%s_epsilon_%0.02f" % ( width, overlap, epsilon )
    plt.savefig( outdir+"%s.png" % ( filename ) )

    output_d3( filename, vertices, edges, "Cover width: %s, Overlap: %s, Epsilon: %0.02f" % ( width, overlap, epsilon ) )


movies = get_movies( movies_dir )

dist_funcs = {}

register_dist_funcs( dist_funcs )

normalize_movies( movies, dist_funcs )

#dispersion_plot( range( 0, len( movies.keys() ) ), dim0, 'movies' )
dispersion_plot( movies )

import pprint
pp = pprint.PrettyPrinter( indent=4 )

# We could in principle have difference means of calculating our
# distance.
distance_func = cartesian_distance
distances = compute_distances( movies, dist_funcs, distance_func )
print "Distances:"
pp.pprint( distances )

#sys.exit( 0 )

#projection_func = eccentricity
#projection_func = density
#projection = compute_projection( distances, projection_func )
#print "Eccentricities:"
#pp.pprint( projection )

epsilon_candidates = cluster_epsilon_finder( movies, distances )
print "Cluster epsilon candidates", epsilon_candidates
epsilon = numpy.median( epsilon_candidates )*1.01
#epsilon = 10
#print "Epsilon selected as: (multiplied by 1.01 to handle rounding errors)", epsilon


f = open( outdir+"graphs.html", 'w' )
html_front = '''
<!DOCTYPE html>
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
<body>
<script src="http://d3js.org/d3.v3.min.js"></script>
<script src="graph.js"></script>
'''
f.write( html_front )
f.close()

'''
Distances:
Cluster epsilon candidates [0.2756963101398022, 0.2824180708397999, 0.2901535353885528, 0.29679948310496235, 0.29809243573819405, 0.3008430147727166, 0.30423158694739116, 0.3111932610996143, 0.3239575458355143, 0.32870830977624693, 0.3349296893257851, 0.339504392995318, 0.34043623376667126, 0.3407374970953071, 0.3456198701639495, 0.3470844154322659, 0.3615455863682823, 0.36873341302556073, 0.3855076149196018, 0.3913397549356695, 0.41500077453947964, 0.4261110496972837, 0.430798082232881, 0.4408081670379248, 0.4411177007877657, 0.4538260438991669, 0.49876468183302686, 0.5161105046270866, 0.5195546689996641, 0.5444156540919776, 0.5715710648582176, 0.5769720423637819, 0.5802541943852065, 0.6011932989376068, 0.6178081743165474, 0.7345168864500814, 0.7384074181643948, 0.8445109840039465, 0.8726441863295533, 0.9951102608430269, 1.016822650925611]
'''

#epsilons = [ 0.32870830977624693, 0.3913397549356695, 0.5444156540919776,  0.9951102608430269 ]

#epsilons = [ 0.35, 0.55 ]

# Decent when including the last metric.
#epsilons = [ .4, .5, .6 ]

epsilons = [ .3, .4, .5 ]

#epsilons = epsilon_candidates

for epsilon in epsilons:

    clusters = get_clusters( movies, distances, epsilon )
    #pp.pprint( clusters )

    movie_clusters = {}
    for movie in movies.keys():
        done = False
        for ( idx, cluster ) in enumerate( clusters ):
            if movie in cluster:
                movie_clusters[movie] = idx
                done = True
                break
        if not done:
            print "ERROR - no cluster idx found for:", movie
            sys.exit( 1 )
    #pp.pprint( movie_clusters )

    cluster_movies = {}
    for ( idx, cluster ) in enumerate( clusters ):
        cluster_movies[idx] = cluster
    pp.pprint( cluster_movies )

#for width in [ .125, .25, float( 1 )/3, .5, float( 2 )/3, 0.75, 0.8]:
#    for slide in [ .5, .6, .7, .8, .9 ]:

#    for width in [ .75, .5, .3 ]:
    for width in [ .45, .6, .75 ]:
        # Slide is intended to be >= 0.5 - by design and conventional
        # practice we don't want more than 2 regions overlapping in one
        # dimension.
        #for slide in [ .9, .7, .5 ]:
        for slide in [ .5, .75, .9 ]:
            # We consider intervals of width whose overlaps are made by
            # sliding the initial point slide percentage of width along.
            #
            # Start at 0 and proceed until we start an
            # interval >=1
            start = 0
            end = 1
            step = width * slide

            movie_cell_map = {}
            cell_movie_map = {}

            #print "Step is:", step

            for movie in sorted( movies.keys() ):
                # We build an array of arrays, the i'th position of which
                # is the array of cells this movie is in for dimension i.
                movie_cells = []

                for dim in dimensions:
                    # Each movie is in 1 or 2 cells for each dimension.
                    dimension_cells = []
                    coordinate = movies[movie][dim]

                    if dim in dist_funcs:
                        value = dist_funcs[dim]( zero[dim], coordinate )
                    else:
                        value = default_dist( zero[dim], coordinate )

                    #print "Coordinate is:", coordinate
                    #print "Value is:", value

                    # Get the latest cell who starts at or before value.
                    first_cell = int( math.floor( float( value ) / step) )

                    # See if this point is also in the prior cell.
                    if first_cell > 0 and step*(first_cell - 1) + width > value:
                        dimension_cells.append( first_cell - 1 )

                    dimension_cells.append( first_cell )
                
                    movie_cells.append( dimension_cells )

                #pp.pprint( movie_cells )

                # Compose a list of cell keys where the cell key is of the
                # form:
                # a-b-c-d...-z where a, b, c are cells this movie is found in.
                next_cell_keys = [ movie_cells[0] ]
                for dim_cells in movie_cells[1:]:
                    cell_keys = next_cell_keys
                    next_cell_keys = []
                    for dim_cell in dim_cells:
                        for cell_key in cell_keys:
                            next_cell_keys.append( cell_key + [dim_cell] )

                cell_keys = next_cell_keys

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
            
            #pp.pprint( cell_movie_map )

            node_map = {}
            movie_node_map = {}
            all_movie_clusters = []
            nodes = []
            edges = []

            # If true only add nodes if they are not strict subsets of existing nodes.
            eliminate_subsets = True

            for cell_string in sorted( cell_movie_map ):
                cell_movie_keys = cell_movie_map[cell_string].keys()
                
                cell_movies = { key : value for ( key, value ) in movies.items() if key in cell_movie_keys }

                cell_movie_clusters = get_clusters( cell_movies, distances, epsilon )

                if eliminate_subsets:
                    for cell_movie_cluster in cell_movie_clusters:
                        cluster_movies = cell_movie_cluster.keys()

                        add_node = True
                        # Determine if this node is a subset of
                        # any existing node.
                        for existing_cluster in all_movie_clusters:
                            subset_of_current = True
                            for cluster_movie in cluster_movies:
                                if cluster_movie not in existing_cluster:
                                    subset_of_current = False
                                    break
                            if subset_of_current:
                                add_node = False
                                break
                        # Delete any existing nodes that are
                        # subsets of this node.
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
                    nodes.append( { "name" : label, 
                                    "group" : movie_clusters[cluster_movies[0]],
                                    "elements" : len( cluster_movies ),
                                    "shading" : 1 } )

                    for cluster_movie in cluster_movies:
                        if cluster_movie not in movie_node_map:
                            movie_node_map[cluster_movie] = { label : True }
                        elif label not in movie_node_map[cluster_movie]:
                            for dest_label in movie_node_map[cluster_movie].keys():
                                edges.append( { "source" : node_map[label],
                                                "target" : node_map[dest_label],
                                                "value" : 1 } )
                        movie_node_map[cluster_movie][label] = True
                        
            filename = "cover_width_%s_overlap_%s_epsilon_%0.02f" % ( width, step, epsilon )

            output_d3( filename, nodes, edges, "Width: %s, Step: %s, Epsilon: %0.02f" % ( width, step, epsilon ) )


#    make_graph( 0, 74, width, 2, epsilon )
#    make_graph( 13000, 33950, width, 4, epsilon )

f = open( outdir+"graphs.html", 'a' )
html_back = '''
</body>
'''
f.write( html_back )
f.close()
