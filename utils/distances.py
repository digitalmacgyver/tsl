#!/usr/bin/env python

import json
import numpy
import os

# Read in movie JSON files.
movies_dir = "../example-scripts/parsed"

def get_movies( movies_dir ):
    '''Returns a hash keyed on movie title whose body is the Python
    data structure made up of the _metrics.json for this film in the
    movies_dir.'''
    movies = {}
    for dirpath, dirnames, filenames in os.walk( movies_dir):
        for directory in dirnames:
            metrics_file = [ x for x in os.listdir( os.path.join( dirpath, directory ) ) if x.endswith( '_metrics.json' ) ][0]
            metrics = json.load( open( os.path.join( dirpath, directory, metrics_file ) ) )
            movies[metrics['title']] = metrics

    return movies

def register_dist_funcs( dist_funcs ):
    
    def mcic( a, b ):
        return abs( a-b )
    dist_funcs[ dimensions[0] ] = mcic

    def poswmc( a, b ):
        return 50*abs( a-b )
    dist_funcs[ dimensions[1] ] = poswmc

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
                dists.append( dist_funcs[dim]( m1[dim], m2[dim] ) )
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
    keys where each hash is a cluster is a subset of the imput movies
    containing the points which are within a transitive closure of
    episolon of one another.'''

    # Don't change the input value.
    movies = {}
    for movie in movies_input.keys():
        movies[movie] = True
    
    clusters = []

    while len( movies ):
        # Avoid iterating over something we're changing.
        movie_keys = movies.keys()
        current_cluster = {}

        for movie in movie_keys:
            if len( current_cluster ) == 0:
                current_cluster[movie] = True
                del movies[movie]
            else:
                for cluster_movie in current_cluster.keys():
                    if distances[cluster_movie][movie] <= epsilon:
                        current_cluster[movie] = True
                        if movie in movies:
                            del movies[movie]
            
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

movies = get_movies( movies_dir )

# Dimensions
#
# Don't change the order of things here unless you also change the
# dist_funcs key lookups in register_dist_funcs

dimensions = [ 'main_character_interlocutor_count', 'percentage_of_scenes_with_main_character' ]

dist_funcs = {}

register_dist_funcs( dist_funcs )

import pprint
pp = pprint.PrettyPrinter( indent=4 )

# We could in principle have difference means of calculating our
# distance.
distance_func = cartesian_distance
distances = compute_distances( movies, dist_funcs, distance_func )
print "Distances:"
pp.pprint( distances )

projection_func = eccentricity
projection = compute_projection( distances, projection_func )
print "Eccentricities:"
pp.pprint( projection )

covering = [ (0, 15), (14, 16), (15, 17), (16, 99) ]

inverse_covering = get_inverse_covering( projection, covering )

epsilon_candidates = cluster_epsilon_finder( movies, distances )
print "Cluster epsilon candidates", epsilon_candidates
epsilon = numpy.median( epsilon_candidates )*1.01
print "Epsilon selected as: (multiplied by 1.01 to handle rounding errors)", epsilon



for partition in inverse_covering:
    partition_clusters = get_clusters( partition['movies'], distances, epsilon )
    print "Range from %s to %s had %s movies, which formed the following clusters:" % ( partition['range'][0], partition['range'][1], len( partition['movies'].keys() ) )
    
    for idx, cluster in enumerate( partition_clusters ):
        print "\tCluster %s" % idx
        for movie in sorted( cluster.keys() ):
            print "\t\t%s" % movie
