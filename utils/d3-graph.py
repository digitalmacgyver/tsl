#!/usr/bin/env python

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

html_back = '''
</body>
'''

html_graph = '''
<p>
Cover: %s, Width: %s, Overlap: %s, Epsilon: %s
</p>
<script>
script_graph( "%s" );
</script>
''' % ( 1, 2, 3, 4, "cover_width_4_overlap_4_epsilon_4.08.json" )

print html_front

print html_graph

html_graph2 = '''
<p>
Cover: %s, Width: %s, Overlap: %s, Epsilon: %s
</p>
<script>
script_graph( "%s" );
</script>
''' % ( 5, 6, 7, 8, "cover_width_4_overlap_4_epsilon_4.08.json" )

print html_graph2

print html_back
