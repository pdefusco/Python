empty_svg = ->
  d3.select('div.output')
    .append('svg')
    .attr("width", 500)
    .attr("height", 666)

shapes = ->
  svg = d3.select('div.output')
    .append('svg')
    .attr("width", 400)
    .attr("height", 300);

  svg.append("circle")
    .attr("class", "myCircles")
    .attr("id", "circle_top")
    .attr("cx", 40)
    .attr("cy", 60)
    .attr("r", 30);

  svg.append("circle")
    .attr("class", "myCircles")
    .attr("id", "circle_bottom")
    .attr("cx", 40)
    .attr("cy", 120)
    .attr("r", 20);

  svg.append("rect")
    .attr("x", 100)
    .attr("y", 60)
    .attr("width", 30)
    .attr("height", 50)
    .style("fill", "orange");

  svg.append("line")
    .attr("x2", 250)
    .attr("y2", 90)
    .attr("x1", 150)
    .attr("y1", 60)
    .attr("stroke", "black")
    .attr("stroke-width", 2);


circles3 = ->
  svg = d3.select('div.output')
    .append('svg')
    .attr("width", 400)
    .attr("height", 300)

  svg.append("circle")
    .attr("cx", 40)
    .attr("cy", 60)
    .attr("r", 10)

  svg.append("circle")
    .attr("cx", 140)
    .attr("cy", 60)
    .attr("r", 10)

  svg.append("circle")
    .attr("cx", 240)
    .attr("cy", 60)
    .attr("r", 10)



rect1 = ->
  svg = d3.select('div.output')
    .append('svg')

  svg.append("rect")
    .attr("x", 150)
    .attr("y", 100)
    .attr("width", 60)
    .attr("height", 300)


rect3 = ->
  svg = d3.select('div.output')
    .append('svg')

  svg.append("rect")
    .attr("x", 200)
    .attr("y", 300)
    .attr("width", 40)
    .attr("height", 50)

  svg.append("rect")
    .attr("x", 100)
    .attr("y", 20)
    .attr("width", 30)
    .attr("height", 50)

  svg.append("rect")
    .attr("x", 10)
    .attr("y", 200)
    .attr("width", 25)
    .attr("height", 90)

# ----------------------------------------------------

slide.title "A basic understanding of JavaScript will save you a lot of pain when you go to build something"


slide.code "JavaScript Basics", null, """
  // Single line comments can be written like this

  /* 
    And multi-line comments like this
  */

  // To print output we use console.log()
  console.log("Hello World!");

  // Whitespace (including newlines) is ignored so..
  console.log("Single line");

  // is the same as
  console
      .log(
               "Not a single line"     );

  // Statements are ended with semi-colons
  console.log("So we can"); console.log("do this");
"""

slide.code "JavaScript Variables: Numbers", null, """
  // Javascript variables are defined with 'var'
  // and are dynamically typed (like python)

  // Their names must start with a alphabetic 
  // character and can consist of integers
  // and underscores

  // There is only one number type (float64)
  var age = 45;

  // There is no integer type, so there is no
  // worry of integer division (python 2)
  // or integer overflow
  console.log("45 = ", 45);
  console.log("45.0 = ", 45.0);

  // Scientific notation is achieved with 'e'
  // xey = x * 10^y
  console.log("1e+1 = ", 1e+1);
  console.log("1e+2 = ", 1e+2);
  console.log("1e-2 = ", 1e-2);

  // Javascript also defines any number larger
  // than 1.7976931348623157e+308 as infinity
  console.log(1.79769313486231570e+308);
  console.log("Anything >= 1.8e+308 = ", 1.8e+308);
"""

slide.code "JavaScript Variables: Strings", null, """
  // Strings work about the same as python
  // You can use " " or '' to define them:
  var name = "Bojack";
  var lastName = 'Horseman';

  // We can concatenate strings with + 
  console.log(name + " " + lastName);

  // And \\ is the escape character 
  console.log(name + " is \\"great\\"");

  // We can also mix quotes for the same effect
  console.log(name + ' is "great"');

  // Multi-line strings require you escape
  // the carridge-return:
  console.log("Multi \\
  .............Line");

  // We can also use + to convert strings to numbers
  var numString = +"55";
  console.log(numString + 1);

  var stringNum = "55";
  console.log(stringNum + 1);

  // We can also check the type of a variable
  // with the typeof operator
  console.log("numString is a " + typeof numString);
  console.log("stringNum is a " + typeof stringNum);
"""

slide.code "JavaScript Arrays", null, """
  // Arrays are squences of elements with integer
  // property names:

  var empty = [];

  var numbers = [
     'zero', 'one', 'two', 'three', 'four',
     'five', 'six', 'seven', 'eight', 'nine'
  ];
  
  console.log("empty[1] = " + empty[1]);
  console.log("numbers[1] = ", numbers[1]);

  // We can also index with strings
  console.log("numbers['1'] = ", numbers['1']);

  // Arrays can hold any mixture of values
  var misc = [
         'string', 98.6, true, false, null, undefined,
         ['nested', 'array'], {object: true}, NaN,
         Infinity
     ];
  
  // Arrays also have a useful length attribute
  console.log("empty.length = " + empty.length);
  console.log("numbers.length = " + numbers.length);
  console.log("misc.length = " + misc.length);
"""


slide.code "JavaScript Array Methods", null, """
// Arrays have a set of methods which are included:
var a = ['a', 'b', 'c'];
var b = ['x', 'y', 'z'];

// Concat:
var c = a.concat(b, true);
console.log("a.concat(b, true) = ", c);

// Pop:
var c = a.pop();
console.log("a.pop() =", c);
console.log("a = ", a);

// Push:
var c = a.push('d');
console.log("a.push() =", c); // new length
console.log("a = ", a);

// Slice:
var a = [1,2,3,4,5,6];
var a_end = a.slice(2);
var a_mid = a.slice(2,4);
console.log("a.slice(2): ", a_end);
console.log("a.slice(2,4): " + a_mid);

// Filter:
var a = [1,4,8,10,3,12,2];
var a_big = a.filter(function (d) {return d > 9;});
console.log("a.filter(function (d) {return d > 9;}) = ", a_big);

// And many more...
// array.reverse()
// array.sort()
// array.shift()
// array.splice()
"""


slide.code "JavaScript Objects", null, """
// Objects are defined with curly braces 
// surrounding zero or more name/value pairs
var empty_object = {};

// Names can be specified with or without quotes
var actor = {
     "first-name": "Bojack",
     last_name: "Horseman"
     };

// Values can be anything, including objects
var course = {
    title: "Data Visualization",
    dept: "DSE",
    number: 241,
    professor: {
        first_name: "Amit",
        last_name: "Chourasia"
      }
  };

// Values are retrieved by wrapping the string
// name in [] or using . notation for legal
// Javascript names
console.log(actor["first-name"], actor.last_name);
console.log(course.title, course.number);

"""


slide.code "JavaScript Objects", null, """
var course = {
    title: "Data Visualization",
    dept: "DSE",
    number: 241,
    professor: {
        first_name: "Amit",
        last_name: "Chourasia"
      }
  };

// Object properties can be replaced
course.title = "Computer Stuff";

// Or if they dont exist, they will be created
course.room = "Dungeon";
console.log("course.room:", course.room);

// Also note objects are always passed by 
// reference and never copied
var course2 = course;
course2.room = "Above Ground!"

console.log("course.room: ", course.room);
console.log("course2.room: ", course2.room);

// Properties can be deleted
delete course.room;
"""

slide.code "Operations and Comparisons", null, """
// Application of operators follows standard
// order of operations:
// . [] ()       Accessing & grouping
// * / %         Mul, Div, Mod
// + -           Add, Subtract
// >= <= > <     Ineqality
// === !==       Equal-to, NEQ
// &&            Logical AND
// ||            Logical OR
// ?:            Ternary
console.log(6*10 + 5 * (2 - 3));

var divByZero = 100.0 / 0.0;
var zeroOverZero = 0.0 / 0.0;

console.log("100.0 / 0.0 = " + divByZero);
console.log("0.0 / 0.0 = " + zeroOverZero);

// Equality returns booleans
console.log("5 === 5: ", 5 === 5);
console.log("5 === 6: ", 5 === 6);
console.log("5 === '5': ", 5 === '5');

// Don't use == for equality...
console.log('\\t\\r\\n ' == 0);

// === checks strick equality
// == performs type conversion 

// Can check for NaN with isNaN()
console.log("isNaN(zeroOverZero):",
              isNaN(zeroOverZero));


"""

slide.code "JavaScript Control Flow", null, """
var a = [0, 1, 2, 3, 4];
var singer = {
    first_name: "Kanye",
    last_name: "West"
}


// If (ALWAYS USE {} BRACES)
if (a[3] === 3) {
  console.log("if: success");
}
else {
  console.log("if: failure");
}

// For (enumeration)
for (var i = 0; i < a.length; i++) {
  console.log("a[" + i + "] = " + a[i]);
}

// For (attr in obj)
for (var prop in singer) {
  console.log("singer[" + prop + "] = ", 
                          singer[prop])
}

// While 
while (a[4] > 2) {
  a[4]--;
  console.log("a[4] = " + a[4]);
}

// Also notice the increment and 
// decrement operators ++, --

"""

slide.code "JavaScript Functions", null, """
// Functions are JavaScript objects

// Create a variable called add and store
// a function in it that adds two numbers.
var add = function (a, b) { 
  return a + b;
};

console.log("add(2,3) = ", add(2,3));

// Since functions are objects, they can be stored
// as 'methods' within other objects

// As Methods, they always recieve the 'this'
// argument when invoked, which is bound 
// to the object they are a method of
var myObject = {
   value: 0,
   increment: function (inc) {
       this.value += inc;
    } 
};

console.log("myObject.value: ", myObject.value);
myObject.increment(2);
console.log("myObject.value: ", myObject.value);
"""

slide.code "JavaScript Functions", null, """
// Functions are JavaScript objects

// Create a variable called add and store
// a function in it that adds two numbers.
var add = function (a, b) { 
  return a + b;
};

// Can also specify function names directly
function sub(a, b) { 
  return a - b;
}; 

console.log("add(2,3) = ", add(2,3));
console.log("sub(2,3) = ", sub(2,3));

// Since functions are objects, they can be stored
// as 'methods' within other objects

// As Methods, they always recieve the 'this'
// argument when invoked, which is bound 
// to the object they are a method of
var myObject = {
   value: 0,
   increment: function (inc) {
       this.value += inc;
    } 
};

console.log("myObject.value: ", myObject.value);
myObject.increment(2);
console.log("myObject.value: ", myObject.value);
"""

slide.title "Theres even more important parts of JavaScript that could help you, like Closures and Inheritance, but for now this is enough. Check out 'JavaScript: The Good Parts' for more."

slide.title "Now some D3..."

slide.code "Selections: d3.select()", null, """
  // d3.select("selector") scans the html document
  // and returns the first instance of 'selector'
  // it finds, where 'selector' is a CSS selector

  // Since these slides have a 
  // <div class="out output"></div>   ----->>>>
  // we select that to work with

  var output_div = d3.select('div.out.output');

  // We can then set CSS style of the selected
  // element with .style('name', value)

  output_div.style('background-color', 'blue');

  // We can modify all other non-style attributes
  // like 'class' and 'id' with .attr()

  output_div.attr('id', 'main_output');

"""


slide.code "Adding DOM elements with D3", null, """
 // First we select the output div 
 var output_div = d3.select('div.output');

 // Use .append() to add a DOM element to 
 // the end of the selected div
 var svg = output_div.append('svg');

 // Use .attr() to set element attributes
 svg.attr("width", 400);
 svg.attr("height", 300);

 // Elements can be added within other elements
 // Add a circle to the svg canvas
 var circle = svg.append("circle");
 circle.attr("cx", 40);
 circle.attr("cy", 60);
 circle.attr("r", 30);

"""


slide.code "Chaining D3 methods", null, """
  // D3 .append(), .attr(), and .style() 
  // all return the element or elements
  // they operated on, so they can be chained
  // as follows:

 var svg = d3.select('div.output')
    .append('svg')
    .attr("width", 400)
    .attr("height", 300);

  // Add a circle
  svg.append("circle")
    .attr("class", "myCircles")
    .attr("id", "circle_top")
    .attr("cx", 40)
    .attr("cy", 60)
    .attr("r", 30);

  // Add second circle
  svg.append("circle")
    .attr("class", "myCircles")
    .attr("id", "circle_bottom")
    .attr("cx", 40)
    .attr("cy", 120)
    .attr("r", 20);

  // Add a rectangle
  svg.append("rect")
    .attr("x", 100)
    .attr("y", 60)
    .attr("width", 30)
    .attr("height", 50)
    .style("fill", "orange");

  // Add a line
  svg.append("line")
    .attr("x2", 250)
    .attr("y2", 90)
    .attr("x1", 150)
    .attr("y1", 60)
    .attr("stroke", "black")
    .attr("stroke-width", 2);
"""

slide.title "Check out the same example <a href=\"./../Basic_Files/basic_d3.html\" >here</a><br/> Right Click and view the source code to understand  how to use d3 inside HTML "
slide.title "Once added, DOM elements can be selected and modified with D3"

slide.code "SVG Selections", shapes, """
 /* Given an output div like:
 <div class="out output">
  <svg height="300" width="400">
    <circle r="30" cy="60" cx="40" 
      id="circle_top" class="myCircles"></circle>
    <circle r="20" cy="120" cx="40"
      id="circle_bottom" class="myCircles"></circle>
    <rect style="fill: orange;" height="50"
       width="30" y="60" x="100"></rect>
    <line stroke-width="2" stroke="black" 
        y1="60" x1="150" y2="90" x2="250"></line>
  </svg>
</div>
 */

 // We can select DOM elements with selector
 // strings (Same as CSS: elem, .class, #id)
 var circle = d3.select("div.output svg")
               .select("#circle_top");

 // We can also sub-select from selections 
 var rect = d3.select("div.output")
              .select("rect");

 // We can then act on these selections
 circle.attr("fill", "red");
 rect.style("fill", "purple");
"""

slide.code "Multiple Selections: .selectAll()", shapes, """
 /* Given an output div like:
 <div class="out output">
  <svg height="300" width="400">
    <circle r="30" cy="60" cx="40" 
      id="circle_top" class="myCircles"></circle>
    <circle r="20" cy="120" cx="40"
      id="circle_bot" class="myCircles"></circle>
    <rect style="fill: orange;" height="50"
       width="30" y="60" x="100"></rect>
    <line stroke-width="2" stroke="black" 
        y1="60" x1="150" y2="90" x2="250"></line>
  </svg>
</div>
 */

 // We can also select ALL elements which match 
 var circle = d3.selectAll(".myCircles");

 // We can then act on all these selections
 // simultaneously
 circle.style("fill", "steelblue");
"""


slide.code "Acting on Selections Individually", shapes, """
/* Given an output div like:
 <div class="out output">
  <svg height="300" width="400">
    <circle r="30" cy="60" cx="40" 
      id="circle_top" class="myCircles"></circle>
    <circle r="20" cy="120" cx="40"
      id="circle_bot" class="myCircles"></circle>
    <rect style="fill: orange;" height="50"
       width="30" y="60" x="100"></rect>
    <line stroke-width="2" stroke="black" 
        y1="60" x1="150" y2="90" x2="250"></line>
  </svg>
</div>
*/

 // Typicall, when selecting SVG elements, we 
 // want to select from the SVG canvas to avoid
 // conflicts
 var svg = d3.select("div.output");

 // Select all circles
 var circle = svg.selectAll("circle");

 // Use an anonymous function which gets
 // evaluated for each element in the selection
 // to set the x coordinate
 circle.attr("cx", function () { 
          return Math.random() * 400;
          });
"""

slide.title "The magic of D3 allows us to then set these element properties based on data"

slide.code "Binding Data: .data()", circles3, """
/* Given an output div like:
<div class="out output">
  <svg height="300" width="400">
    <circle r="10" cy="60" cx="40"></circle>
    <circle r="10" cy="60" cx="140"></circle>
    <circle r="10" cy="60" cx="240"></circle>
  </svg>
</div>
*/

// Select the SVG canvas
var svg = d3.select("div.output svg");

// Select all three circles
var circle = svg.selectAll("circle");

// Define our data
var dataset = [25, 400, 900];

// Bind the circles to data of our choice
// based on index
circle.data(dataset);

// After the data is bound, it lives
// in the .__data__ property of the 
// DOM element (CHECK INSPECTOR)

// This data is then available as the 
// first argument to .attr() and .style()
// functions (by convention we use d)
circle.attr("r", function (d) { 
        return Math.sqrt(d);
        });
"""

slide.code "Binding Data: .data()", circles3, """
/* Given an output div like:
<div class="out output">
  <svg height="300" width="400">
    <circle r="10" cy="60" cx="40"></circle>
    <circle r="10" cy="60" cx="140"></circle>
    <circle r="10" cy="60" cx="240"></circle>
  </svg>
</div>
*/

 // Select the SVG canvas
 var svg = d3.select("div.output svg");

 // Select all three circles
 var circle = svg.selectAll("circle");

 // Define the data
 var dataset = [25, 400, 900];

 // Bind the circles to data of our choice
 // based on index
 circle.data(dataset);

 // After the data is bound, it lives
 // in the .__data__ property of the 
 // DOM element (CHECK INSPECTOR)

 // This data is then available as the 
 // first argument to .attr() and .style()
 // functions (by convention we use d)
 circle.attr("r", function (d) { 
          return Math.sqrt(d);
          });

 // The second argument is the index of the 
 // element (by convention we use i)
 circle.attr("cy", function (d, i) {
      return i * 100 + 150;
  });
"""

slide.title "We can even create new elements for new data"

slide.code "Entering Elements: .enter()", circles3, """
/* Given an output div like:
<div class="out output">
  <svg height="300" width="400">
    <circle r="10" cy="60" cx="40"></circle>
    <circle r="10" cy="60" cx="140"></circle>
    <circle r="10" cy="60" cx="240"></circle>
  </svg>
</div>
*/

 // What if we try to bind 4 data points
 // instead of 3 like last time?
 var dataset = [25, 400, 900, 1600];

 // Select the SVG canvas
 var svg = d3.select("div.output svg");

 // Select all three circles & bind to
 // our four data points
 var circle = svg.selectAll("circle")
              .data(dataset);

 // Change radius of existing circles based on data
 circle.attr("r", function (d) { 
          return Math.sqrt(d);
      })

 // Then use .enter() to create & select
 // placeholder elements for which we have data 
 // but no existing element (datapoint 1600)
 var circleEnter = circle.enter();

 // Add a circle for each new data point
 var newCircles = circleEnter.append("circle");

 // Now set the properties of the new circle
 newCircles.attr("r", function (d) { 
          return Math.sqrt(d);
      })
      .attr("cx", function (d, i) {
          return i * 100 + 40;
        })
      .attr("cy", 60)
      .style("fill", "red");
"""


slide.code "ENTER EVERYTHING", empty_svg, """
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="666" width="500"></svg>
</div>
*/

// We can now create elements for all new data

// Select the SVG canvas
var svg = d3.select("div.output svg");

var dataset = [25, 400, 900, 1600];

// Select circles & bind to
// our four data points
var circle = svg.selectAll("circle")
            .data(dataset);

// Entering now returns all data
var circleEnter = circle.enter();

// Add a circle for each new data point
var newCircles = circleEnter.append("circle");

// Now set the properties of the new circles
newCircles.attr("r", function (d) { 
        return Math.sqrt(d);
    })
    .attr("cx", function (d, i) {
        return i * 100 + 40;
    })
    .attr("cy", 60);
"""


slide.code "ENTER EVERYTHING: The pattern", empty_svg, """
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="666" width="500"></svg>
</div>
*/

// Select the SVG canvas
var svg = d3.select("div.output svg");

// Define the data
var dataset = [25, 400, 900, 1600];

// Bind - Enter - Update
svg.selectAll("circle")
    .data(dataset)
    .enter()
    .append("circle")
    .attr("r", function (d) { 
      return Math.sqrt(d);
    })
    .attr("cx", function (d, i) {
        return i * 100 + 40;
      })
    .attr("cy", 60);
"""

slide.title "And remove elements for missing data"

slide.code "Exiting Elements: .exit()", circles3, """
/* Given an output div like:
<div class="out output">
  <svg height="300" width="400">
    <circle r="10" cy="60" cx="40"></circle>
    <circle r="10" cy="60" cx="140"></circle>
    <circle r="10" cy="60" cx="240"></circle>
  </svg>
</div>
*/

// What if we instead give the visualization
// only 2 data points?
var dataset = [25, 400];

// Select the SVG canvas
var svg = d3.select("div.output svg");

// Select all three circles & bind to
// our two data points
var circle = svg.selectAll("circle")
            .data(dataset);

// Change radius of existing circles based on data
circle.attr("r", function (d) { 
        return Math.sqrt(d);
    })

// Here .exit() returns the elements for which
// no corresponding data point was bound.
circle.exit().remove();

// If the data doesnt exist, remove the circle!
"""


slide.title "To assist visualization, D3 provides scales to convert from a specified domain to a specified range"

slide.code "Scales", null, """
// Given a canvas width and height
var w = 420, h = 320;

// We use d3.scaleLinear() to 
// return a function which converts
// from our data domain to the 
// canvas pixel domain

// x is a function!
var x = d3.scaleLinear()
  // Domain is input
  .domain([-1, 1])
  // Range is output
  .range([0, w])

// y is also a function!
var y = d3.scaleLinear()
  .domain([0, 1000])
  .range([0, h])

console.log("x(-1) ==", x(-1)) // == 0
console.log("x(0) ==", x(0)) // == w/2
console.log("x(1) ==", x(1)) // == w

console.log("y(900) ==", y(900)) // == h*9/10

"""

slide.code "A basic scaled barchart", empty_svg, """
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="666" width="500"></svg>
</div>
*/

var svg = d3.select("div.output svg");

var data = [4, 8, 15, 16, 23, 42];

var width = 500,
    height = 666;

// Define the bar thickness to be an even
// division of the svg height
var barHeight = height / data.length;

// Create an xScale which maps from data
// values to x coordinates
var xScale = d3.scaleLinear()
    .domain([0, d3.max(data)])
    .range([0, width]);

// For each data point, create a group
// which translates all elements to the 
// correct y coordinate
var bar = svg.selectAll("g")
    .data(data)
  .enter().append("g")
    .attr("transform", function(d, i) { 
      return "translate(0," + i * barHeight + ")"; 
    });

// bar now holds all newly created groups

// Add an SVG rect of width = xScale(datum)
bar.append("rect")
    .attr("width", function (d) { 
      return xScale(d); 
    })
    .attr("height", barHeight - 1)
    .style("fill", "steelblue")
    .style("stroke", "white");

// Add SVG text the end of the bar displaying
// the value of the data
bar.append("text")
    .attr("x", function(d) { 
        return xScale(d) - 5; 
    })
    .attr("y", barHeight / 2)
    .attr("dy", ".35em")
    .text(function(d) { return d; })
    .style("fill", "white")
    .style("text-anchor", "end");
"""

slide.code "Flipping the barchart", empty_svg, """
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="666" width="500"></svg>
</div>
*/

var svg = d3.select("div.output svg");

var data = [4, 8, 15, 16, 23, 42];

var width = 500,
    height = 666,
    barWidth = width / data.length - 10;

var y_scale = d3.scaleLinear()
    .domain([0, d3.max(data)])
    .range([0, height]);

var x_scale = d3.scaleLinear()
    .domain([0, data.length])
    .range([0, width]);

// Create a bar for each data point
var bar = svg.selectAll("rect")
    .data(data)
  .enter().append("rect")
    .attr("x", function (d, i) { 
      return x_scale(i); 
    })
    .attr("y", function (d) { 
      return height - y_scale(d); 
    })
    .attr("width", barWidth)
    .attr("height", function (d) { 
      return y_scale(d); 
    })
    .style("fill", "steelblue")
    .style("stroke", "white");

// Create a text label for each data point
svg.selectAll("text")
    .data(data)
    .enter().append("text")
    .attr("x", function(d, i) { 
        return x_scale(i) + barWidth / 2; 
    })
    .attr("y", function (d) { 
      return height - y_scale(d) + 20; 
    })
    .text(function(d) { return d; })
    .style("fill", "white")
    .style("text-anchor", "middle");
"""

slide.code "Basic Scatterplot", empty_svg, """
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="666" width="500"></svg>
</div>
*/

// Select the svg canvas
var svg = d3.select("div.output svg");

var data = 
  [[5, 20], [480, 90], [250, 50], [100, 33], [330, 95],
   [410, 12], [475, 44], [25, 67], [85, 21], [220, 88]];

var width = 500,
    height = 666;

// Compute the maximum values for the scales
var x_max = d3.max(data, function(d) {
        return d[0];
    });

var y_max = d3.max(data, function (d) {
      return d[1];
    });

// Define two linear scales for the 
// x & y values
var x_scale = d3.scaleLinear()
    .domain([0, x_max])
    .range([0, width]);

var y_scale = d3.scaleLinear()
    .domain([0, y_max])
    .range([0, height]);

// Bind - enter - update circles with
// coordinates given by the scaled data
svg.selectAll("circle") 
   .data(data)
   .enter()
   .append("circle")
   .attr("cx", function(d) {
        return x_scale(d[0]);
   })
   .attr("cy", function(d) {
        return y_scale(d[1]);
   })
   .attr("r", 5);
"""

slide.code "Basic Histogram", empty_svg ,"""
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="666" width="500"></svg>
</div>
*/
//generate random data
var data = d3.range(1000).map(d3.randomBates(10));

var formatCount = d3.format(",.0f");

//Initialize width, height. Select SVG and add svg group "g"
var svg = d3.select("div.output svg"),
    margin = {top: 20, right: 30, bottom: 30, left: 30},
    width = 500 - margin.left - margin.right,
    height = +svg.attr("height") - margin.top - margin.bottom,
    g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

//Define XScales
var x = d3.scaleLinear()
    .rangeRound([0, width]);

//Define Histogram bins
var bins = d3.histogram()
    .domain(x.domain())
    (data);

//Define YScales using bins
var y = d3.scaleLinear()
    .domain([0, d3.max(bins, function(d) { return d.length; })])
    .range([height, 0]);

//Generate Histogram
var bar = g.selectAll(".bar")
  .data(bins)
  .enter().append("g")
    .attr("class", "bar")
    .attr("transform", 
      function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; });

bar.append("rect")
    .attr("x", 1)
    .attr("width", x(bins[0].x1) - x(bins[0].x0) - 1)
    .attr("height", function(d) { return height - y(d.length); })
    .style("fill","lightblue");

bar.append("text")
    .attr("dy", "-.75em")
    .attr("y", 6)
    .attr("x", (x(bins[0].x1) - x(bins[0].x0)) / 2)
    .attr("text-anchor", "middle")
    .text(function(d) { return formatCount(d.length); });

g.append("g")
    .attr("class", "axis axis--x")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));
"""


slide.title "D3 also makes animations between properties easy"

slide.code "Basic Transition", circles3, """
/* Given an output div like:
<div class="out output">
  <svg height="300" width="400">
    <circle r="10" cy="60" cx="40"></circle>
    <circle r="10" cy="60" cx="140"></circle>
    <circle r="10" cy="60" cx="240"></circle>
  </svg>
</div>
*/

 // Select the SVG canvas
 var svg = d3.select("div.output svg");

// Select our circles and bind new data
var circle = svg.selectAll("circle")
            .data([25, 400, 900]);

// Apply transition of duration 2 seconds
// and update properties of all circles 
// based on new data
circle.transition()
      .duration(2000)
      .attr("r", function (d) { 
        return Math.sqrt(d);
      })
      .attr("cx", function (d, i) {
          return i * 100 + 40;
      })
      .attr("cy", function (d, i) {
          return i * 100 + 150;
      });
"""

slide.code "Transitions For New Elements", circles3, """
/* Given an output div like:
<div class="out output">
  <svg height="300" width="400">
    <circle r="10" cy="60" cx="40"></circle>
    <circle r="10" cy="60" cx="140"></circle>
    <circle r="10" cy="60" cx="240"></circle>
  </svg>
</div>
*/

// Select the SVG canvas
 var svg = d3.select("div.output svg");

// Select our circles and bind new data
var circle = svg.selectAll("circle")
            .data([25, 400, 900, 1600]);

// Define initial properties for new circle
// Apply transition of duration 2 seconds
// and update properties of all circles 
// based on data
circle.enter()
  .append("circle")
  .attr("r", 0)
  .attr("cx", 400)
  .attr("cy", 300)
  .style("fill", "red")
      .merge(circle) 

      // allows transition to be applied on the newly entered circle 
          .transition()
          .duration(2000)
          .attr("r", function (d) { 
            return Math.sqrt(d);
          })
          .attr("cy", function (d, i) {
              return i * 100 + 150;
          });
"""

slide.code "Transitioning in the barchart", empty_svg, """
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="666" width="500"></svg>
</div>
*/

var svg = d3.select("div.output svg");

var data = [4, 8, 15, 16, 23, 42];

var width = 500,
    height = 666,
    barWidth = width / data.length - 10;

var y_scale = d3.scaleLinear()
    .domain([0, d3.max(data)])
    .range([0, height]);

var x_scale = d3.scaleLinear()
    .domain([0, data.length])
    .range([0, width]);

// Create a bar for each data point
var bar = svg.selectAll("rect")
    .data(data)
  .enter().append("rect")
    .attr("x", function (d, i) { 
      return x_scale(i); 
    })
    .attr("y", function (d) { 
      return height; 
    })
    .attr("width", barWidth)
    .attr("height", function (d) { 
      return y_scale(d); 
    })
    .style("fill", "steelblue")
    .style("stroke", "white");

// Transition to the correct y location
bar.transition().duration(2000)
    .attr("y", function (d) { 
      return height - y_scale(d); 
    });

// Create a text label for each data point
var text = svg.selectAll("text")
    .data(data)
    .enter().append("text")
    .attr("x", function(d, i) { 
        return x_scale(i) + barWidth / 2; 
    })
    .attr("y", height)
    .text(function(d) { return d; })
    .style("fill", "white")
    .style("text-anchor", "middle");

// Transition text to correct height
text.transition().duration(2000)
    .attr("y", function (d) { 
      return height - y_scale(d) + 20; 
    });
"""



# -----------------------------------------------
slide.title "Binding data by key"

slide.code_title title = ".data(..., join)"

init_svg = ->
  svg = d3.select("div.output").append("svg")

  svg.selectAll("rect")
    .data([127, 61, 256])
    .enter().append("rect")
      .attr("x", 0)
      .attr("y", (d,i) -> i*90+50)
      .attr("width", (d,i) -> d)
      .attr("height", 20)
      .style("fill", "steelblue")

slide.code title, init_svg, """
var svg = d3.select("div.output svg")

// Let's say we start here:
/*
svg.selectAll("rect")
  .data([127, 61, 256])
  .enter().append("rect")
    .attr("x", 0)
    .attr("y", function(d,i) { return i*90+50 })
    .attr("width", function(d,i) { return d; })
    .attr("height", 20)
    .style("fill", "steelblue")
*/

// And then we bind new data by index
var selection = svg.selectAll("rect")
  .data([61, 256, 71]) 

// Create rectangles for new data (NONE)
selection.enter().append("rect")
  .attr("x", 0)
  .attr("y", function(d,i) { return i*90+50 })
  .attr("width", function(d,i) { return d; })
  .attr("height", 20)
  .style("fill", "steelblue")
// Transition old rectangles to new y and 
// width based on the new data
    .merge(selection)
      .transition()
      .duration(3000)
        .attr("x", 0)
        .attr("y", function(d,i) { return i*90+50 })
        .attr("width", function(d,i) { return d; })
        .attr("height", 20)
        .style("fill", "steelblue")

// Remove rectangles for which we have no bound
// data (NONE since we bound by index)
selection.exit()
  .remove()
"""



# -----------------------------------------------
slide.title "Loading External Data"

slide.code "d3.csv()", empty_svg, """
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="666" width="500"></svg>
</div>
*/

/* And a datafile data.csv:
      name,value
      Locke,4
      Reyes,8
      Ford,15
      Jarrah,16
      Shephard,23
      Kwon,42
*/

var width = 450,
    height = 666;

var x = d3.scaleLinear()
    .range([0, width]);

var chart = d3.select("div.output svg");

// d3.csv takes 3 arguments:
// filename, accessor, callback

// The accessor is a function which takes 
// each row of the data and returns a converted
// version:
function type(d) {
  d.value = +d.value; // coerce to number
  return d;
}

// This converted version is passed to the
// third argument, the callback function
// All processing happens inside this function

d3.csv("data/data.csv", type,
  function(error, data) {
  // Stop d3.csv() from failing to load silently
  if (error) { console.log(error);}

  var barHeight = height / data.length;

  x.domain([0, d3.max(data, function(d) {
     return +d.value; 
     })]);

  chart.attr("height", barHeight * data.length);

  var bar = chart.selectAll("g")
      .data(data)
    .enter().append("g")
      .attr("transform", function(d, i) { 
          return "translate(0," + i * barHeight + ")";
        });

  bar.append("rect")
      .attr("width", function(d) { 
          return x(d.value); 
       })
      .attr("height", barHeight - 1);

  bar.append("text")
      .attr("x", function(d) { 
          return x(d.value) - 25; 
      })
      .attr("y", barHeight / 2)
      .attr("dy", ".35em")
      .text(function(d) { 
        return d.value; 
      });
});

"""

# -----------------------------------------------
slide.title "D3 Data Processing"

slide.code "d3.nest()", null, """
// Given an yields of items
var yields = [
  {yield: 27.00, variety: "Manchuria", year: 1931, site: "University Farm"},
  {yield: 48.87, variety: "Manchuria", year: 1931, site: "Waseca"},
  {yield: 27.43, variety: "Manchuria", year: 1932, site: "Morris"},
  {yield: 27.43, variety: "Chowmein", year: 1932, site: "Morris"},
  {yield: 27.00, variety: "Manchuria", year: 1932, site: "University Farm"},
  {yield: 48.87, variety: "Dumpling", year: 1933, site: "Waseca"},
  {yield: 27.43, variety: "Manchuria", year: 1933, site: "Morris"},
  {yield: 27.43, variety: "Chowmein", year: 1935, site: "Morris"}
];

//Nesting allows elements in an array to 
//be grouped into a hierarchical tree structure; 
// Think of it like the GROUP BY operator in SQL
var entries = d3.nest()
    .key(function(d) { return d.year; })
    .entries(yields);

//Print JSON in pretty format
console.log(JSON.stringify(
                 entries,undefined,2))
"""

slide.code "d3.nest() rollup", null, """
// Given an yields of items
var yields = [
  {yield: 27.00, variety: "Manchuria", year: 1931, site: "University Farm"},
  {yield: 48.87, variety: "Manchuria", year: 1931, site: "Waseca"},
  {yield: 27.43, variety: "Manchuria", year: 1932, site: "Morris"},
  {yield: 27.43, variety: "Chowmein", year: 1932, site: "Morris"},
  {yield: 27.00, variety: "Manchuria", year: 1932, site: "University Farm"},
  {yield: 48.87, variety: "Dumpling", year: 1933, site: "Waseca"},
  {yield: 27.43, variety: "Manchuria", year: 1933, site: "Morris"},
  {yield: 27.43, variety: "Chowmein", year: 1935, site: "Morris"}
];

//Nesting allows elements in an array to 
//be grouped into a hierarchical tree structure; 
// Think of it like the GROUP BY operator in SQL
var entries = d3.nest()
    .key(function(d) { return d.year; })
    .key(function(d) { return d.variety; })
    .rollup(function(d) { return d.length; })
    .entries(yields);

//Print JSON in pretty format
console.log(JSON.stringify(
                 entries,undefined,2))
"""

slide.title "Check more on nesting <a target=\"_blank\" href=\"https://github.com/d3/d3-collection#nests\">here</a>"
# -----------------------------------------------
slide.title "Advanced Scales"

slide.code "Advanced Scales", empty_svg, """
var w = 450,
    h = 500;

var years = ["1992", "1996", "2000", "2004"];
var positions = [100, 200, 300, 400];

// Ordinal scales map discrete values by index
var xScale = d3.scaleOrdinal()
                .domain(years)
                .range(positions);

console.log("xScale('2000'): ", xScale("2000"));

// ScaleBands are like ordinal scales except 
// the output range is continuous and numeric.
// rangeRound() create the range by diving the given interval into 
// bands of even size (with rounded values)

var xScale =  d3.scaleBand()
                .domain(years)
                .rangeRound([0, w])
                .padding(0.05)

console.log("xScale('2000'): ", xScale("2000"));

// There is also a .bandwidth() method which 
// returns the size of the band
console.log("xScale.bandwidth(): ", 
              xScale.bandwidth());
"""

# -----------------------------------------------
slide.title "Adding Axes"

slide.code "Adding Axes", empty_svg, """
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="600" width="500"></svg>
</div>
*/

// Select the SVG canvas
var svg = d3.select('div.output svg');

// Given a canvas width and height
var w = 500, h = 600, padding = 100;

// First get the scales
var xScale = d3.scaleLinear()
  .domain([-1, 1])
  .range([0, w])

var yScale = d3.scaleLinear()
  .domain([0, 1000])
  .range([0, h])

/* Define the axes using axes
orientation and scale functions
For orientation we use:
axisLeft,axisRight,axisTop or axisBottom
*/
var xAxis = d3.axisBottom(xScale);  
var yAxis = d3.axisLeft(yScale);

// To actually draw the SVG axis to the screen
// we have to say 'where' and give it something
// to be drawn in (like a <g> tag). We then use
// .call() to call the Axis drawing functions
svg.append("g")
    .attr("transform", "translate(0," + h + ")")
    .call(xAxis);

svg.append("g")
    .attr("transform", "translate(" + padding + ", 0)")
    .call(yAxis);

"""

# -----------------------------------------------
slide.title "Interesting Examples"



slide.code "Force-Directed Graph", empty_svg, """
/* Given a div with an empty SVG canvas:
<div class="out output">
    <svg height="600" width="500"></svg>
</div>
*/

var svg = d3.select("div.output").select("svg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

var color = d3.scaleOrdinal(d3.schemeCategory20);

var simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(function(d) { return d.id; }))
    .force("charge", d3.forceManyBody())
    .force("center", d3.forceCenter(width / 2, height / 2));

d3.json("data/miserables.json", function(error, graph) {
  if (error) throw error;

  var link = svg.append("g")
      .attr("class", "links")
    .selectAll("line")
    .data(graph.links)
    .enter().append("line")
      .attr("stroke-width", 
              function(d) { return Math.sqrt(d.value); })
      .attr("stroke","#999");

  var node = svg.append("g")
      .attr("class", "nodes")
    .selectAll("circle")
    .data(graph.nodes)
    .enter().append("circle")
      .attr("r", 5)
      .attr("fill", function(d) { return color(d.group); })
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended));

  node.append("title")
      .text(function(d) { return d.id; });

  simulation
      .nodes(graph.nodes)
      .on("tick", ticked);

  simulation.force("link")
      .links(graph.links);

  function ticked() {
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  }
});

function dragstarted(d) {
  if (!d3.event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(d) {
  d.fx = d3.event.x;
  d.fy = d3.event.y;
}

function dragended(d) {
  if (!d3.event.active) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}
"""

slide.code "Expandible Dendrite Plot", null, """
var treeData =
  {
    "name": "Top Level",
    "children": [
      { 
        "name": "Level 2: A",
        "children": [
          { "name": "Son of A" },
          { "name": "Daughter of A" }
        ]
      },
      { "name": "Level 2: B" }
    ]
  };

// Set the dimensions and margins of the diagram
var margin = {top: 20, right: 90, bottom: 30, left: 90},
    width = 380 - margin.left - margin.right,
    height = 666 - margin.top - margin.bottom;

// append the svg object to the body of the page
// appends a 'group' element to 'svg'
// moves the 'group' element to the top left margin
var svg = d3.select("div.output").append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate("
          + margin.left + "," + margin.top + ")");

var i = 0,
    duration = 750,
    root;

// declares a tree layout and assigns the size
var treemap = d3.tree().size([height, width]);

// Assigns parent, children, height, depth
root = d3.hierarchy(treeData, function(d) { return d.children; });
root.x0 = height / 2;
root.y0 = 0;

// Collapse after the second level
root.children.forEach(collapse);

update(root);

// Collapse the node and all it's children
function collapse(d) {
  if(d.children) {
    d._children = d.children
    d._children.forEach(collapse)
    d.children = null
  }
}

function update(source) {

  // Assigns the x and y position for the nodes
  var treeData = treemap(root);

  // Compute the new tree layout.
  var nodes = treeData.descendants(),
      links = treeData.descendants().slice(1);

  // Normalize for fixed-depth.
  nodes.forEach(function(d){ d.y = d.depth * 180});

  // ****************** Nodes section ***************************

  // Update the nodes...
  var node = svg.selectAll('g.node')
      .data(nodes, function(d) {return d.id || (d.id = ++i); });

  // Enter any new modes at the parent's previous position.
  var nodeEnter = node.enter().append('g')
      .attr('class', 'node')
      .attr("transform", function(d) {
        return "translate(" + source.y0 + "," + source.x0 + ")";
    })
    .on('click', click);

  // Add Circle for the nodes
  nodeEnter.append('circle')
      .attr('class', 'node')
      .attr('r', 1e-6)
      .style("fill", function(d) {
          return d._children ? "lightsteelblue" : "#fff";
      })
      .attr("stroke","#ccc");

  // Add labels for the nodes
  nodeEnter.append('text')
      .attr("dy", ".35em")
      .attr("x", function(d) {
          return d.children || d._children ? -13 : 13;
      })
      .attr("text-anchor", function(d) {
          return d.children || d._children ? "end" : "start";
      })
      .text(function(d) { return d.data.name; });

  // UPDATE
  var nodeUpdate = nodeEnter.merge(node);

  // Transition to the proper position for the node
  nodeUpdate.transition()
    .duration(duration)
    .attr("transform", function(d) { 
        return "translate(" + d.y + "," + d.x + ")";
     });

  // Update the node attributes and style
  nodeUpdate.select('circle.node')
    .attr('r', 10)
    .style("fill", function(d) {
        return d._children ? "lightsteelblue" : "#fff";
    })
    .attr('cursor', 'pointer');


  // Remove any exiting nodes
  var nodeExit = node.exit().transition()
      .duration(duration)
      .attr("transform", function(d) {
          return "translate(" + source.y + "," + source.x + ")";
      })
      .remove();

  // On exit reduce the node circles size to 0
  nodeExit.select('circle')
    .attr('r', 1e-6);

  // On exit reduce the opacity of text labels
  nodeExit.select('text')
    .style('fill-opacity', 1e-6);

  // ****************** links section ***************************

  // Update the links...
  var link = svg.selectAll('path.link')
      .data(links, function(d) { return d.id; });

  // Enter any new links at the parent's previous position.
  var linkEnter = link.enter().insert('path', "g")
      .attr("class", "link")
      .attr('d', function(d){
        var o = {x: source.x0, y: source.y0}
        return diagonal(o, o)
      }).attr("stroke","#ccc") 
        .attr("stroke-width","2px")
        .attr("fill","None")


  // UPDATE
  var linkUpdate = linkEnter.merge(link);

  // Transition back to the parent element position
  linkUpdate.transition()
      .duration(duration)
      .attr('d', function(d){ return diagonal(d, d.parent) });

  // Remove any exiting links
  var linkExit = link.exit().transition()
      .duration(duration)
      .attr('d', function(d) {
        var o = {x: source.x, y: source.y}
        return diagonal(o, o)
      })
      .remove();

  // Store the old positions for transition.
  nodes.forEach(function(d){
    d.x0 = d.x;
    d.y0 = d.y;
  });

  // Creates a curved (diagonal) path from parent to the child nodes
  function diagonal(s, d) {

    var path = `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
              ${(s.y + d.y) / 2} ${d.x},
              ${d.y} ${d.x}`

    return path
  }

  // Toggle children on click.
  function click(d) {
    if (d.children) {
        d._children = d.children;
        d.children = null;
      } else {
        d.children = d._children;
        d._children = null;
      }
    update(d);
  }
}
"""

