var BOUNDARY_VERTICAL = "V";
var BOUNDARY_HORIZONTAL = "H";

var MINIMAL_PADDING = 0.1;
var MAXIMAL_PADDING = 0.4;
if (MAXIMAL_PADDING>0.5) { throw "Maximal padding must be <= 0.5" }

function isBoundaryIndexValid(b, N) { 
	if (b.type===BOUNDARY_VERTICAL) { 
		if (b.row<0 || b.row>=N || b.col<0 || b.col>=N-1) { 
			return false;
		}
	} else if (b.type === BOUNDARY_HORIZONTAL) { 
		if (b.row<0 || b.row>=N-1 || b.col<0 || b.col>=N) { 
			return false;
		}
	} 
	return true;
}

function isBoundaryIndexValidWithEdges(b, N) { 
	if (b.type===BOUNDARY_VERTICAL) { 
		if (b.row<0 || b.row>=N || b.col<-1 || b.col>=N) { 
			return false;
		}
	} else if (b.type === BOUNDARY_HORIZONTAL) { 
		if (b.row<-1 || b.row>=N || b.col<0 || b.col>=N) { 
			return false;	
		}
	} 
	return true;
}


var Point = function(grid,row, col) { 
	this.row = row;
	this.col = col;
	this.grid = grid;

	if ((this.row<-1 || this.row>this.grid.N-1 || this.col<-1 || this.col>this.grid.N-1)) { 
		throw "Illegal point ("+p+") for N="+N;
	}

	this.traverse = function(b) {
		var edges = [] 
		if (b.type===BOUNDARY_HORIZONTAL) { 
			edges.push(this.grid.Point(b.row, b.col-1));
			edges.push(this.grid.Point(b.row, b.col));
		} else if (b.type===BOUNDARY_VERTICAL) { 
			edges.push(this.grid.Point(b.row-1,b.col));
			edges.push(this.grid.Point(b.row, b.col));
		}

		var that = this;
		if (!edges.some(function(x) { return that.equals(x) })) { 
			throw "Boundary "+b+" doesn't touch point "+that;
		}
		return edges.filter(function(x) { return !that.equals(x) })[0];
	}
}
Point.prototype.equals = function(other) { 
	return this.row === other.row && this.col === other.col;
}
Point.prototype.toString = function() { 
	return "|"+this.row+","+this.col+"|";
}




var Boundary = function(grid, type, row, col) { 
	this.type = type;
	this.row = row;
	this.col = col;
	this.grid = grid;

	this.validate = function() { 
		if (this.type !== BOUNDARY_HORIZONTAL && this.type !== BOUNDARY_VERTICAL) { 
			throw "Illegal type " + type + " for Boundary";
		}
		if (!isBoundaryIndexValidWithEdges(this, this.grid.N)) { 
			throw "Illegal boundary index ("+this.row+","+this.col+") for N="+this.grid.N+"("+this.type+")";
		}
		return true;
	}

	this.getTwoPoints = function() { 
		if (this.type===BOUNDARY_HORIZONTAL) { 
			return [ this.grid.Point(this.row, this.col-1), this.grid.Point(this.row, this.col) ];
		} else if (this.type === BOUNDARY_VERTICAL) { 
			return [ this.grid.Point(this.row-1, this.col),  this.grid.Point(this.row, this.col) ];
		}
	}

	this.meetingPoint = function(other) { 
		var myPoints = this.getTwoPoints();
		var otherPoints = other.getTwoPoints();
		var meetingPoint = myPoints.filter(function(p) { return otherPoints.some(function(p2) { return p.equals(p2); } ) } )[0];
		return meetingPoint;
	}

	// headPoint gives us the "direction" in which we are traveling. this is important to set the padding internally.
	this.randomizePoint = function(headPoint) { 
		var p = null;
		var randOffset = Math.random()*(1-2*MAXIMAL_PADDING)+MAXIMAL_PADDING;
		if (this.type === BOUNDARY_HORIZONTAL) { 
			p = this.grid.Point(this.row, this.col - randOffset);
		} else if (this.type === BOUNDARY_VERTICAL) { 
			p = this.grid.Point(this.row - randOffset, this.col);
		}

		// adding internal padding
		var paddingAmount = Math.random()*(MAXIMAL_PADDING-MINIMAL_PADDING)+MINIMAL_PADDING;
		var isHead = (headPoint.row===this.row && headPoint.col===this.col);
		if (this.type === BOUNDARY_VERTICAL && isHead) { 
			p.col -= paddingAmount;
		} else if (this.type === BOUNDARY_VERTICAL) { 
			p.col += paddingAmount;
		} else if (this.type === BOUNDARY_HORIZONTAL && isHead) { 
			p.row += paddingAmount;
		} else {
			p.row -= paddingAmount;
		} 
		return p;
	}

};
Boundary.prototype.equals = function(other) { 
	return this.type===other.type && this.row===other.row && this.col===other.col;
};
Boundary.prototype.toString = function() { 
	return "["+this.type+","+this.row+","+this.col+"]";
}



var Cell = function(grid, row, col) { 
	this.row = row;
	this.col = col;
	this.grid = grid;
};
Cell.prototype.equals= function(other) { 
	return this.row===other.row && this.col===other.col;
};
Cell.prototype.toString = function() { 
	return "("+this.row+","+this.col+")";
}



var Grid = function(n) { 
	this.Point = function(row, col) { 
		return new Point(this,row, col);
	}
	this.Cell = function(row, col) { 
		return new Cell(this, row, col);
	}
	this.Boundary = function(type, row, col) { 
		return new Boundary(this, type, row, col);
	}

	var initialize_boundaries = function(N) { 
		// initialize all boundaries
		var boundaries = {};
		boundaries[BOUNDARY_VERTICAL] = {};
		boundaries[BOUNDARY_HORIZONTAL] = {};
		// vertical ones
		for (var row=0 ; row<N ; row++) { 
			boundaries[BOUNDARY_VERTICAL][row] = {};

			for (var col=-1 ; col<N ; col++) {
				boundaries[BOUNDARY_VERTICAL][row][col] = 1;
			}
		}
		// horizontal ones
		for (var row=-1 ; row<N ; row++) { 
			boundaries[BOUNDARY_HORIZONTAL][row] = {};

			for (var col=0 ; col<N ; col++) { 
				boundaries[BOUNDARY_HORIZONTAL][row][col] = 1;
			}
		}
		return boundaries;
	}

	var initialize_matrix = function(N) { 
		var mat = [];
		for (var i=0 ; i<N ; i++) { 
			mat[i] = [];
		}
		for (var i=0 ; i<N*N ; i++) { 
			mat[Math.floor(i/N)][i%N] = i;
		}
		return mat;
	}

	// setup the object properties
	this.N = n;
	this.boundaries = initialize_boundaries(this.N);
	this.matrix = initialize_matrix(this.N);



	function numRowsCols(type, N) { 
		if (type===BOUNDARY_HORIZONTAL) { 
			return [N-1, N];
		} else if (type === BOUNDARY_VERTICAL) { 
			return [N, N-1];
		}
	}


	// some methods
	this.dropBoundary = function(b)  {
		// validateBoundaryType(b.type);
		// validateBoundaryIndicies(b);

		var cell1;
		var cell2;
		if (b.type===BOUNDARY_VERTICAL) { 
			cell1 = this.matrix[b.row][b.col];
			cell2 = this.matrix[b.row][b.col+1]; 
		} else if (b.type === BOUNDARY_HORIZONTAL) { 
			cell1 = this.matrix[b.row][b.col];
			cell2 = this.matrix[b.row+1][b.col];
		} 

		// replacing cell2 with cell1
		for (var i=0 ; i<this.N ; i++) { 
			for(var j=0 ; j<this.N ; j++) { 
				if (this.matrix[i][j] === cell2) { 
					this.matrix[i][j] = cell1;
				}
			}
		}

		this.boundaries[b.type][b.row][b.col] = 0;
	}

	this.apply_boundary_pct = function(pct) { 
		for (var row=0 ; row<this.N ; row++) { 
			for (var col=0 ; col<this.N-1 ; col++) {
				if (Math.random()<1-pct) { 
					this.dropBoundary(this.Boundary(BOUNDARY_VERTICAL, row, col));
				}
			}
		}
		// horizontal ones
		for (var row=0 ; row<this.N-1 ; row++) { 
			for (var col=0 ; col<this.N ; col++) { 
				if (Math.random()<1-pct) { 
					this.dropBoundary(this.Boundary(BOUNDARY_HORIZONTAL, row, col));
				}
			}
		}
	}

	this.getBoundariesStrByType = function(type) { 
		// validateBoundaryType(type);

		var res = []

		var rowsCols = numRowsCols(type, this.N);
		var numRows = rowsCols[0];
		var numCols = rowsCols[1];

		for (var i=0 ; i<numRows ; i++) { 
			var row = "";
			for (var j=0 ; j<numCols ; j++) { 
				if (this.boundaries[type][i][j] === 1) { 
					row += "*";
				} else { 
					row += " ";
				}
			}
			res.push(row);
		}
		return res;
	};


	this.getBoundariesStr = function() { 
		return this.getBoundariesStrByType(BOUNDARY_VERTICAL) + "\n" + this.getBoundariesStrByType(BOUNDARY_HORIZONTAL);
	}

	this.adjacentCellsForBoundary = function(b) { 
		// validateBoundaryType(b.type);
		// validateBoundaryIndiciesWithEdges(b, this.N);

		var res = [];
		if (b.type === BOUNDARY_HORIZONTAL) { 
			res.push(this.Cell(b.row, b.col));
			res.push(this.Cell(b.row+1, b.col));
		} else if (b.type === BOUNDARY_VERTICAL) { 
			res.push(this.Cell(b.row, b.col));
			res.push(this.Cell(b.row, b.col+1));
		}

		// filter only valid cells
		var rowsCols = numRowsCols(b.type, this.N);
		var numRows = rowsCols[0];
		var numCols = rowsCols[1];
		
		var that = this;
		res = res.filter(function(x) { 
			return (x.row>=0 && x.row<that.N && x.col>=0 && x.col<that.N);
		});

		return res;
	}


	// this method actually enforces clock-wise traversal of the graph (it returns the boundaries in anti-clockwise direction)
	this.allBoundariesOrdered = function(p, startingFrom) { 
		// validatePoint(p, this.N);

		var res = [];
		res.push(this.Boundary(BOUNDARY_VERTICAL, p.row, p.col));
		res.push(this.Boundary(BOUNDARY_HORIZONTAL, p.row, p.col));
		res.push(this.Boundary(BOUNDARY_VERTICAL, p.row+1, p.col));
		res.push(this.Boundary(BOUNDARY_HORIZONTAL, p.row, p.col+1));

		if (startingFrom) { 
			var foundStartingFrom = false;
			for(var i=0 ; i<4 ; i++) { 
				if (res[i].equals(startingFrom)) { foundStartingFrom = true; break; }
			}
			if (!foundStartingFrom) { throw "StartingFrom (" + startingFrom +") wasn't found in boundaries (" + res + ")"; }
			res = res.slice(i+1).concat(res.slice(0,i+1));
		}

		var that = this;
		res = res.filter(function(b) { 
			return isBoundaryIndexValidWithEdges(b, that.N);
		});

		return res;
	}


	this.findPathForCell = function(cell) { 
		if (this.boundaries[BOUNDARY_HORIZONTAL][cell.row-1][cell.col] !== 1) {
			throw "Cell " + cell + " doesn't have a top boundary in place";
		}

		var cellValue = this.matrix[cell.row][cell.col];

		var currP = this.Point(cell.row-1, cell.col);
		var currBoundary = this.Boundary(BOUNDARY_HORIZONTAL, cell.row-1, cell.col);
		var startingBoundary = currBoundary;
		
		var path = [ ];

		var that = this;
		do { 
			path.push(currBoundary.randomizePoint(currP));

			var possibleBoundaries = this.allBoundariesOrdered(currP, currBoundary)
			// boundary exists
			.filter(function(x) { 
				return that.boundaries[x.type][x.row][x.col]===1;
			})
			// one of its sides is the original cell
			.filter(function(x) { 
				return that.adjacentCellsForBoundary(x).some(function(y) { return that.matrix[y.row][y.col]===cellValue; });
			});
			
			if (possibleBoundaries.length === 0) { 
				throw "Stuck with path: " + path;
			}

			// identify non-convex turns
			if (!this.isConvexTurn(currBoundary, possibleBoundaries[0], cell)) { 
				var meetingPoint = currBoundary.meetingPoint(possibleBoundaries[0]);
				path.push(meetingPoint);
			}

			var currBoundary = possibleBoundaries[0];

			currP = currP.traverse(currBoundary);
		} while (!currBoundary.equals(startingBoundary))

		return path.concat(path[0]);
	}

	this.cellBoundaryDirection = function(cell, boundary) {
		if (boundary.type === BOUNDARY_HORIZONTAL && cell.row-1===boundary.row && cell.col === boundary.col) { 
			return 1;
		} else if (boundary.type===BOUNDARY_HORIZONTAL && cell.row===boundary.row && cell.col===boundary.col) { 
			return 3;
		} else if (boundary.type===BOUNDARY_VERTICAL && cell.row===boundary.row && cell.col===boundary.col) { 
			return 2;
		} else if (boundary.type===BOUNDARY_VERTICAL && cell.row===boundary.row && cell.col-1===boundary.col) {
			return 4;
		} else { 
			throw "Boundary " + boundary + " and cell " + cell +" doens't match";
		}
	}

	this.isConvexTurn = function(b1, b2, cell) { 
		// find meeting point
		var meetingPoint = b1.meetingPoint(b2);
		if (!meetingPoint) { throw "Boundaries " + b1 + " and " + b2 + " doesn't meet"; }

		if (b1.type===b2.type) { return true; }

		var b1Cells = this.adjacentCellsForBoundary(b1);
		var b2Cells = this.adjacentCellsForBoundary(b2);
		var sharedCell = b1Cells.filter(function(c) { return b2Cells.some(function(c2) { return c.equals(c2);  } ) } )[0];

		if (!sharedCell) { throw "Boundaries " + b1 + " and " + b2 + " doesn't have a shared cell"; }

		return this.matrix[sharedCell.row][sharedCell.col] === this.matrix[cell.row][cell.col];
	}

	this.findCellsStartingPoints = function() { 
		var res = {};
		for (var row=0; row<this.N; row++) { 
			for(var col=0 ; col<this.N; col++) { 
				var cellValue = this.matrix[row][col];
				if (res[cellValue]) { continue; }

				// find a cell with top border in this block
				while (row > 0 && this.matrix[row-1][col]===cellValue) { 
					row -= 1;
				}
				var cell = this.Cell(row, col);

				res[cellValue] = cell;
			}
		}
		return res;
	}

	this.getAllPaths = function() { 
		var cells = this.findCellsStartingPoints();
		for (var c in cells) { 
			var cell = cells[c];
			// var p = this.Point(cell.row-1, cell.col);

			cells[c] = this.findPathForCell(cell);
		}
		return cells;
	}
};





if (typeof exports !== 'undefined') { 
	exports.Grid = Grid;
}