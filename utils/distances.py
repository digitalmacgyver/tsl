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

movies = get_movies( movies_dir )

# Dimensions
#
# Don't change the order of things here unless you also change the
# dist_funcs key lookups in register_dist_funcs

dimensions = [ 'main_character_interlocutor_count', 'precentage_of_scenes_with_main_character' ]

dist_funcs = {}

def register_dist_funcs( dist_funcs ):
    
    def mcic( a, b ):
        return abs( a-b )
    dist_funcs[ dimensions[0] ] = mcic

    def poswmc( a, b ):
        return 50*abs( a-b )
    dist_funcs[ dimensions[1] ] = poswmc

register_dist_funcs( dist_funcs )

def cartesian_distance( dists ):
    '''Takes in an array of distances between coordinates, and
    aggregates them into a single distance function.  Here we use
    Cartesian distance.'''
    total_dist = 0
    for dist in dists:
        total_dist += dist**2
    return total_dist**0.5

# We could in principle have difference means of calculating our
# distance.
distance_func = cartesian_distance

def compute_distances( movies, dist_funcs ):
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
    result = {}
    denominator = len( distances.keys() )
    for k1 in sorted( distances.keys() ):
        numerator = 0
        for k2, distance in distances[k1].items():
            numerator += distance
        result[k1] = numerator / denominator
    return result

projection_func = eccentricity

def compute_projection( distances, projection_func ):
    return projection_func( distances )


import pprint
pp = pprint.PrettyPrinter( indent=4 )

distances = compute_distances( movies, dist_funcs )

print "Distances:"
pp.pprint( distances )

projection = compute_projection( distances, projection_func )

print "Eccentricities:"
pp.pprint( projection )

