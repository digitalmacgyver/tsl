function script_graph( filename, width=512, height=288, show_labels=false ) {
    var color = d3.scale.category20();

    var force = d3.layout.force()
	.charge(-60)
	.linkDistance(30)
	.size([width, height]);

    var svg = d3.select("body").append("svg")
	.attr("width", width)
	.attr("height", height);

    d3.json( filename, function(error, graph) {
	force
	    .nodes(graph.nodes)
	    .links(graph.links)
	    .start();

	var link = svg.selectAll(".link")
	    .data(graph.links)
	    .enter().append("line")
	    .attr("class", "link")
	    .style("stroke-width", function(d) { return Math.sqrt(d.value); });

	var node = undefined;

	if ( show_labels ) {
	    var gnodes = svg.selectAll("g.gnode")
	        .data( graph.nodes )
	        .enter().append( "g" )
	        .classed( "gnode", true );

	    node = gnodes.append( "circle" )
		.attr( "class", "node" )
	        .attr( "r", 5 )
		.style( "fill", function( d ) { return color( d.group ); } )
		.call( force.drag );
	    
	    var labels = gnodes.append( "text" )
		.text( function( d ) { return d.name } );
	} else {
	    var node = svg.selectAll(".node")
		.data(graph.nodes)
		.enter().append("circle")
		.attr("class", "node")
		.attr("r", 5)
		.style("fill", function(d) { return color(d.group); })
		.call(force.drag);

	    node.append("title")
		.text(function(d) { return d.name; });
	}	    

	force.on("tick", function() {
	    link.attr("x1", function(d) { return d.source.x; })
		.attr("y1", function(d) { return d.source.y; })
		.attr("x2", function(d) { return d.target.x; })
		.attr("y2", function(d) { return d.target.y; });

	    if ( show_labels ) {
		gnodes.attr( "transform", function( d ) {
		    return 'translate(' + [d.x, d.y] + ')';
		} );
	    } else {
		node.attr("cx", function(d) { return d.x; })
		    .attr("cy", function(d) { return d.y; });
	    }
	});
    });
}
