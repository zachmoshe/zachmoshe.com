'use strict';


/**
 * @ngdoc overview
 * @name randomPolygonsApp
 * @description
 * # randomPolygonsApp
 *
 * Main module of the application.
 */
var app = angular.module('randomPolygonsApp', []);

app	.controller('polygonsCtrl', ['$scope', function($scope) { 
	$scope.gridSizes = [10,25,50,75];
	$scope.grid_size = 50;
	$scope.grid_pct = 0.75;

	$scope.estAvgBlockSize = function(n,p) { 
		n = parseFloat(n)
		p = parseFloat(p)
		if (p===0) { 
			return n*n;
		}
    		return Math.pow( ( Math.pow(1-p,n)*(n*p-p+2)+p-2 ) / p, 2)
	};

	$scope.getBlockExampleId = function(avg_block_size) { 
		if (avg_block_size < 1.5) { 
			return 0;
		} else if (avg_block_size < 2.5) { 
			return 1;
		} else if (avg_block_size < 3.5) { 
			return 2;
		} else if (avg_block_size < 5) { 
			return 3;
		} else if (avg_block_size < 8) { 
			return 4;
		} else if (avg_block_size < 15) { 
			return 5;
		} else if (avg_block_size < 35) { 
			return 6;
		} else { 
			return 7;
		}
	};

}]);



app.directive('polygonsCanvas', function () {
    return {
      template: '',
      restrict: 'A',

      link: function postLink($scope, $element, $attrs) {
        // setup
        $element.attr('height', '10000');
        $element.attr('width', '10000');

        // generate the first grid
        $scope.reset($scope.grid_size, $scope.grid_pct);
      },
      
      controller: [ '$scope', '$element', function($scope, $element) { 

        var canvas = $element[0]
        var context = canvas.getContext('2d');


        $scope.reset = function(n,pct) { 
		this.N = n;
		$scope.grid = new Grid(n);
		$scope.grid.apply_boundary_pct(pct);

		$scope.drawGrid();
        }

        $scope.drawGrid = function() { 
        	var grid = $scope.grid;

		$scope.width = canvas.width;
		$scope.height = canvas.height;
		$scope.cellHeight = $scope.height / grid.N;
		$scope.cellWidth = $scope.width / grid.N;

		clearBoard();
		drawPolygons("black");
		//drawBoundariesOld("#BBBBBB")
        };

        function clearBoard(width, height) { 
		// clear canvas
		context.clearRect(0,0, $scope.width, $scope.height);

		// draw the border
		context.beginPath();
		context.lineWidth=2;
		context.strokeStyle = "black";
		context.rect(0,0, $scope.width, $scope.height);
		context.stroke();
        }

        function drawPolygons(color) { 
        	var grid = $scope.grid;

        	context.lineWidth = 1;
    		context.strokeStyle = color;

		var paths = grid.getAllPaths();
		for (var cellValue in paths) { 
        		var path = paths[cellValue];

        		context.beginPath();
        		for (var i=0 ; i<path.length; i++) {
        			var p = path[i];
        			context.lineTo($scope.cellWidth*(p.col+1), $scope.cellHeight*(p.row+1));
        		}
        		context.fillStyle = color;
        		context.fill();
        		context.stroke();
        	}
        }

	function drawBoundariesOld(color) { 
		var grid = $scope.grid;

		// draw all boundaries
		context.lineWidth = 1;
		for (var row=0 ; row<grid.N ; row++) { 
			for (var col=0 ; col<grid.N ; col++) { 
				row = parseInt(row)
				col = parseInt(col)

				if (grid.boundaries["V"][row][col] === 1) { 
					context.beginPath();
					context.strokeStyle = color;
					context.moveTo($scope.cellWidth*(col+1), $scope.cellHeight*row);
					context.lineTo($scope.cellWidth*(col+1), $scope.cellHeight*(row+1));
					context.stroke();
				}
			}
		}
		for (var row=0; row<grid.N ; row++) { 
			for (var col=0 ; col<grid.N; col++) { 
				row = parseInt(row);
				col = parseInt(col);

				if (grid.boundaries["H"][row][col] === 1)  {
					context.beginPath();
					context.strokeStyle = color;
					context.moveTo($scope.cellWidth*col, $scope.cellHeight*(row+1));
					context.lineTo($scope.cellWidth*(col+1), $scope.cellHeight*(row+1));
					context.stroke();
				}
			}
		}
        }

      }]

    };
  });






