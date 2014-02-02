#!/usr/bin/env python

import json
import os

# Read in movie JSON files.
movies_dir = "../example-scripts/parsed"

def get_movies( movies_dir ):
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
    '''Hash of hash, keys are two movies, value is distance between
    them.'''
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
                
def get_clusters( movies_input, distances, threshold ):
    '''Given a hash of movie keys, the distances data structure, and a
    threshold distance, returns an array of hashes of movie keys where
    each hash is a cluster.'''

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
                    if distances[cluster_movie][movie] < threshold:
                        current_cluster[movie] = True
                        if movie in movies:
                            del movies[movie]
            
        clusters.append( current_cluster )

    return clusters

movies = get_movies( movies_dir )

# Dimensions
#
# Don't change the order of things here unless you also change the
# dist_funcs key lookups in register_dist_funcs

dimensions = [ 'main_character_interlocutor_count', 'precentage_of_scenes_with_main_character' ]

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

for partition in inverse_covering:
    partition_clusters = get_clusters( partition['movies'], distances, 10 )
    print "Range from %s to %s had %s movies, which formed the following clusters:" % ( partition['range'][0], partition['range'][1], len( partition['movies'].keys() ) )
    
    for idx, cluster in enumerate( partition_clusters ):
        print "\tCluster %s" % idx
        for movie in sorted( cluster.keys() ):
            print "\t\t%s" % movie

