function script_graph( filename, element, width, height, show_labels  ) {
    var color = d3.scale.category20();
    var clique_color = d3.scale.category10();
    
    var force = d3.layout.force()
	.charge( -60 )
	.linkDistance( 150 )
	.friction( .75 )
	.size( [ width, height ] );
    
    var drag = force.drag().on( "dragstart", dragstart );
    
    var svg = d3.select( element ).append("svg")
	.attr("width", width)
	.attr("height", height);
    
    d3.json( filename, function(error, graph) {
	force
	    .nodes(graph.nodes)
	    .links(graph.links)
	    .start();
	
	// Do not reorder the definition of cliques, nodes, and links,
	// as this interferes with which elements are "on top" in the
	// SVG drawing.
	var clique_nodes = graph.cliques.map( function ( d ) { return d.map( function ( e ) { return graph.nodes[e]; } ) } );
	
	var clique_path = function( d ) {
	    return "M" + d3.geom.hull( d.map( function ( i ) { return [ i.x, i.y ]; } ) ).join( "L" ) + "Z";
	};
	
	var clique_fill = function( d ) { 
	    if ( d.length == 4 ) {
		return "#8856a7";
	    }
	    if ( d.length == 3 ) {
		return "#9ebcda";
	    }
	};
	//var clique_fill = function ( d, i ) { return color( i ); };
	
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
		.on( "dblclick", dblclick )
		.call( drag );
	    
	    var labels = gnodes.append( "text" )
		.text( function( d ) { return d.name } );
	} else {
	    var node = svg.selectAll(".node")
		.data(graph.nodes)
		.enter().append("circle")
		.attr("class", "node")
		.attr("r", function( d ) { return 4*Math.sqrt( d.elements ); } )
		.style("fill", function(d) { return "black"; 
					     //return color(d.group); 
					   })
		.style("fill-opacity", function( d ) { return d.shading; } )
		.on( "dblclick", dblclick )
		.call( drag );
	    
	    node.append("title")
		.text(function(d) { return d.name; });
	}	    
	
	force.on("tick", function() {
	    link.attr("x1", function(d) { return d.source.x; })
		.attr("y1", function(d) { return d.source.y; })
		.attr("x2", function(d) { return d.target.x; })
		.attr("y2", function(d) { return d.target.y; });
	    
	    var clique_nodes = graph.cliques.map( function ( d ) { return d.map( function ( e ) { return graph.nodes[e]; } ) } );
	    
	    svg.selectAll( "path" )
		.data( clique_nodes )
		.attr( "d", clique_path )
		.enter().insert( "path", "circle" )
		.style( "fill", clique_fill )
		.style( "stroke", clique_fill )
		.style( "stroke-width", 15 )
		.style( "stroke-linejoin", "miter" )
		.style( "opacity", .4 )
		.attr( "d", clique_path );
	    
	    if ( show_labels ) {
		gnodes.attr( "transform", function( d ) {
		    return 'translate(' + [d.x, d.y] + ')';
		} );
	    } else {
		
		node.attr( "transform", function( d ) {
		    return 'translate(' + [d.x, d.y] + ')';
		} );
		//node.attr("cx", function(d) { return d.x; })
		//.attr("cy", function(d) { return d.y; });
	    }
	});
    });
    
    function dblclick( d ) {
	d3.select( this ).classed( "fixed", d.fixed = false );
    }
	
    function dragstart( d ) {
	d3.select( this ).classed( "fixed", d.fixed = true );
    }
}
