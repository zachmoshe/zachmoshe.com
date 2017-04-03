var rp = require('./random-polygons-common.js')

try {
	var grid = new rp.Grid(5)
	grid.apply_boundary_pct(0.7)

	console.log(grid.matrix);
	console.log(grid.boundaries);
	console.log();
	console.log(grid.getBoundariesStr());
	console.log();

	var c = grid.Cell(3,2);
	var p = grid.Point(3,2);
	var b = grid.Boundary('V',3,2);
	var b2 = grid.Boundary('H',2,3);
	// console.log(grid.isConvexTurn(b,b2, c));

	console.log(grid.findPathForCell(c) + " ");
	var paths = grid.getAllPaths();
	for (var c in paths) { 
		var path = paths[c];
		console.log(c + " - " + path);
	}




} catch(err) { 
	console.log(err);
	console.log(err.stack);
}
