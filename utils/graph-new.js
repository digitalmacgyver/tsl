function script_graph( filename, width=724, height=408, show_labels=false ) {
    var color = d3.scale.category20();
    var clique_color = d3.scale.category10();

    var force = d3.layout.force()
	.charge(-60)
	.linkDistance(60)
	.size([width, height]);

    var svg = d3.select("body").append("svg")
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

	var lineFunction = d3.svg.line()
	  .x(function(d) { return d.x; })
	  .y(function(d) { return d.y; })
	  .interpolate("linear");

	var cliques = svg.selectAll( ".clique" )
	  .data( clique_nodes )
	  .enter().append( "path" )
	  .attr( "class", "clique" )
	  .attr("d", function( d ) { return lineFunction( d ) + "Z" } )
	    //.attr("stroke", "blue")
	    //.attr("stroke-width", 2)
	  .attr("fill", function( d ) { 
	      if ( d.length == 4 ) {
		  return "#8856a7";
	      }
	      if ( d.length == 3 ) {
		  return "#e0ecf4";
	      }
	      return clique_color( d.length );
	   } )
	  .style( "fill-opacity", 0.5 );

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
	    .attr("r", function( d ) { return 4*Math.sqrt( d.elements ); } )
	    .style("fill", function(d) { return "black"; 
		//return color(d.group); 
	      })
	    .style("fill-opacity", function( d ) { return d.shading; } )
	    .call(force.drag);

	    node.append("title")
		.text(function(d) { return d.name; });
	}	    

	force.on("tick", function() {
	    link.attr("x1", function(d) { return d.source.x; })
		.attr("y1", function(d) { return d.source.y; })
		.attr("x2", function(d) { return d.target.x; })
		.attr("y2", function(d) { return d.target.y; });
	    
	    var clique_nodes = graph.cliques.map( function ( d ) { return d.map( function ( e ) { return graph.nodes[e]; } ) } );

	    svg.selectAll( ".clique" )
	      .data( clique_nodes )
	      .attr("d", function( d ) { return lineFunction( d ) + "Z" } );
	    
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
}
