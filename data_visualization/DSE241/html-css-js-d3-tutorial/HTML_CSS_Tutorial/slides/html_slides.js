// Generated by CoffeeScript 1.12.2
(function() {
  var circles3, empty_svg, rect1, rect3, shapes;

  empty_svg = function() {
    return d3.select('div.output').append('svg').attr("width", 500).attr("height", 666);
  };

  shapes = function() {
    var svg;
    svg = d3.select('div.output').append('svg').attr("width", 400).attr("height", 300);
    svg.append("circle").attr("class", "myCircles").attr("id", "circle_top").attr("cx", 40).attr("cy", 60).attr("r", 30);
    svg.append("circle").attr("class", "myCircles").attr("id", "circle_bottom").attr("cx", 40).attr("cy", 120).attr("r", 20);
    svg.append("rect").attr("x", 100).attr("y", 60).attr("width", 30).attr("height", 50).style("fill", "orange");
    return svg.append("line").attr("x2", 250).attr("y2", 90).attr("x1", 150).attr("y1", 60).attr("stroke", "black").attr("stroke-width", 2);
  };

  circles3 = function() {
    var svg;
    svg = d3.select('div.output').append('svg').attr("width", 400).attr("height", 300);
    svg.append("circle").attr("cx", 40).attr("cy", 60).attr("r", 10);
    svg.append("circle").attr("cx", 140).attr("cy", 60).attr("r", 10);
    return svg.append("circle").attr("cx", 240).attr("cy", 60).attr("r", 10);
  };

  rect1 = function() {
    var svg;
    svg = d3.select('div.output').append('svg');
    return svg.append("rect").attr("x", 150).attr("y", 100).attr("width", 60).attr("height", 300);
  };

  rect3 = function() {
    var svg;
    svg = d3.select('div.output').append('svg');
    svg.append("rect").attr("x", 200).attr("y", 300).attr("width", 40).attr("height", 50);
    svg.append("rect").attr("x", 100).attr("y", 20).attr("width", 30).attr("height", 50);
    return svg.append("rect").attr("x", 10).attr("y", 200).attr("width", 25).attr("height", 90);
  };

  slide.title("HyperText Markup Language (HTML) is the standard markup language for creating web pages and web applications. ");

  slide.title("A basic understanding of HTML is essential to build web based visualizations");

  slide.code("HTML Elements", null, "<!--comments can be written like this\n   The whole paragraph is commented out -->\n <!--\n   An HTML element usually consists of a start tag \n   and end tag, with the content inserted in between:\n\n   <tagname>Content goes here...</tagname>\n\n   Few essential tags are\n   <html> … </html>\n   <head> … </head>\n   <body> … </body>\n\n   Few common tags are\n   <div> … </div>\n   <img> … </img>\n   <p> … </p>\n   <script> … </script>\n   <table> … </table>\n   <ul> … </ul>\n\n   Below are some examples on  their usage\n -->\n\n \n");

  slide.code("HTML Elements", null, "\n<h1>This is a Heading</h1>\n\n  <h2>This is another Heading</h2>\n\n  <!-- There are other types of headings such as h3, h4, h5 and h6 -->\n\n  <p>This is a  paragraph.</p>\n\n  <a href=\"http://www.google.com\" \n  target=\"_blank\">This is a link</a>\n\n  <img src=\"data/logo_UCSD.png\"/>\n\n  <!-- Check out div and span. \n  These are called Block Elements -->");

  slide.code("HTML Document", null, " \n\n<!-- A simple HTML Document looks like this -->\n\n<!DOCTYPE html>\n<html>\n  <head>\n    <title>Page Title</title>\n  </head>\n  <body>\n\n    <h1>My First Heading</h1>\n    <p>My first paragraph.</p>\n\n  </body>\n</html>");

  slide.code("HTML Attributes", null, "\n<!-- Attributes provide additional \n  information about HTML elements.-->\n\n  <img src=\"data/logo_UCSD.png\" \n    width=\"250\" height=\"142\">\n\n  <p title=\"I'm a tooltip\">\n  This is a paragraph.\n  </p>");

  slide.code("Styling", null, "<!-- HTML allows inline styling \n  of elements using \"style\"\n  attribute. -->\n<!-- Use INSPECTOR to know values for each attribute -->\n\n<h1 style=\"color:blue;\n    font-family:verdana;\">\n    This is a heading\n</h1>\n<p style=\"color:red;\n    font-size:160%;   \n    text-align:center;\n    background-color:yellow\">\n  This is a paragraph.\n</p>\n\n<!-- For text formatting,\n  We can use special elements\n-->\n\n<strong>This text is strong</strong>\n\n<b>This text is bold</b>\n\n<i>This text is italic</i>\n\n<h2>HTML <small>Small</small> Formatting</h2>\n\n<h2>HTML <mark>Marked</mark> Formatting</h2>");

  slide.code("Tables", null, "<!-- Table is an important and very useful\nelement. It can be used to effectively display information\nas well as to create a layout of the HTML page -->\n\n<table style=\"width:100%;\" border=1px\">\n  <tr>\n    <th>Firstname</th>\n    <th>Lastname</th> \n    <th>Age</th>\n  </tr>\n  <tr>\n    <td>Firstname 1</td>\n    <td>Lastname 1</td> \n    <td>24</td>\n  </tr>\n  <tr>\n    <td>Firstname 2</td>\n    <td>Lastname 2</td> \n    <td>94</td>\n  </tr>\n</table>");

  slide.code("Lists", null, "\n<!-- List is a collection of related items.\n     It is another effective way to display information -->\n\n<!-- UNORDERED LISTS -->\n<ul style=\"list-style-type:disc\">\n  <li>Coffee</li>\n  <li>Tea</li>\n  <li>Milk</li>\n</ul>\n\n<!-- ORDERED LISTS -->\n<ol>\n  <li>Coffee</li>\n  <li>Tea</li>\n  <li>Milk</li>\n</ol>\n\n<!-- NESTED LISTS -->\n<ul>\n  <li>Coffee</li>\n  <li>Tea\n    <ul>\n    <li>Black tea</li>\n    <li>Green tea</li>\n    </ul>\n  </li>\n  <li>Milk</li>\n</ul>");

  slide.code("Blocks", null, "<!-- <div> element is often used as\n a container for other HTML elements -->\n\n<div style=\"background-color:black;color:white;padding:20px;\">\n  <h2>Outer Div</h2>\n  <p>This is the outer div element</p>\n  <div style=\"background-color:yellow;color:Black;\">\n    <h2> Inner Div </h2>\n    <p> This is the inner div </p>\n  </div>\n\n</div>\n\n");
  
  slide.code("Radio and Checkboxes", null, '<!-- Radio and Checkboxes -->\n\n<h1>Radio</h1>\n\n<input type="radio" name="gender" \n value="male" checked> Male<br> \n\n<input type="radio" name="gender" \nvalue="female"> Female<br> \n\n<input type="radio" name="gender" \nvalue="other"> Other \n\n\n\n<h1>Checkbox</h1>\n<input type="checkbox" name="vehicle" \nvalue="Bike">I have a bicyle<br>\n\n<input type="checkbox" name="vehicle" \nvalue="Bike">I have a bike<br>\n\n<input type="checkbox" name="vehicle" \nvalue="Car">I have a car<br>');
  
  slide.code('Dropdown', null, '<h1>Dropdown</h1>\n<select>\n  <option value="volvo">Volvo</option>\n  <option value="saab">Saab</option>\n  <option value="mercedes">Mercedes</option>\n  <option value="audi">Audi</option>\n</select>');

  slide.code("Range Slider", null, '<h1>Range Slider</h1>\n\n<input type="range" min="1" max="100" value="50">');
  
  slide.title('<a href="https://www.w3schools.com/jquerymobile/tryit.asp?filename=tryjqmob_forms_slider_range">Double Range Slider</a>');

  slide.title("Let's look a bit Javascript. JavaScript is used with HTML to make dynamic changes to the webpage ");

  slide.code("The script Tag", null, "<!-- The <script> tag is used to define a \n    client-side script (JavaScript) -->\n\n<!-- Lets extend our basic HTML example\n      to include <script>-->\n <!DOCTYPE html>\n  <html>\n    <head>\n      <title>Page Title</title>\n    </head>\n    <body>\n\n      <h1>My First Heading</h1>\n      <p>My first paragraph.</p>\n     <script>\n      // This is a javascript comment.\n      </script>\n    </body>\n  </html>\n\n\n");

  slide.code("id and class attributes", null, "<!-- id is used to select one\n representative element -->\n\n<p id=\"demo\">E0</p>\n<script>\ndocument.getElementById(\"demo\").innerHTML \n= \"Hello JavaScript!\";\n</script> \n\n<!-- class is used to select a collection\nof elements -->\n<p class=\"demo_class\">E1</p>\n<p class=\"demo_class\">E2</p>\n<p class=\"demo_class\">E3</p>\n<p class=\"demo_class2\">E4</p>\n<script>\nvar elements = document.getElementsByClassName\n              (\"demo_class\");\nfor(var i=0; i<elements.length; i++)\n  elements[i].style.color = \"red\";;\n</script> \n\n\n<!-- Check out \n  document.getElementsByName\n  document.getElementsByTagName\n-->\n");

  slide.code("JavaScript Selectors", null, "\n<div>\n\n  <h2> div  </h2>\n  <div id=\"demo\"> \n    <h2> div 2 </h2>\n    <div>\n      <h2> div 3 </h2>\n    </div>\n  </div>\n</div>\n\n<script>\n  var el = document.getElementById(\"demo\");\n\n  el.parentNode.style.backgroundColor   = \"red\";\n  el.getElementsByTagName(\"div\")[0].style\n  .backgroundColor = \"yellow\"\n  el.children[0].style.color = \"indigo\"\n</script>\n\n");

  slide.code("JavaScript Events", null, "\n<div id=\"demo\">\n\n  <h2> div  </h2>\n  <div > \n    <h2> div 2 </h2>\n    <div>\n      <h2> div 3 </h2>\n    </div>\n  </div>\n</div>\n\n<script>\n  document.getElementById(\"demo\")\n    .addEventListener(\"mouseover\", \n      function(){\n        this.style.backgroundColor = \"red\"\n      });\n  document.getElementById(\"demo\")\n    .addEventListener(\"click\", \n      function(){\n        this.style.backgroundColor = \"yellow\"\n      });\n  document.getElementById(\"demo\") \n    .addEventListener(\"mouseout\", \n      function(){\n        this.style.backgroundColor = \"cyan\"\n      });\n</script>\n\n");

  slide.code("Simple Calculator", null, "<div id=\"calculator\">\n   <h1 id=\"screen\">0</h1>\n   <div id=\"numpad\">\n     <ul>\n       <li><button class=\"key\" onclick=\"add(1)\">1</button></li>\n       <li><button class=\"key\" onclick=\"add(2)\">2</button></li>\n       <li><button class=\"key\" onclick=\"add(3)\">3</button></li>\n       <li><button class=\"key\" onclick=\"add(4)\">4</button></li>            \n   </div>\n</div>\n<script>\nvar sum = 0;\nfunction add(x){\n  document.getElementById(\"screen\").innerText = (sum+x)\n  sum += x;\n}\n</script>");

  slide.title("A separate tutorial covers the Javascript language in depth <a href=\"../../D3_tutorial/slides/index.html\">here</a>");

  slide.title("Lets dive into styling using CSS");

  slide.code("CSS Basics", null, "<!-- \n  We  generally declare the style element\n  inside the <head> \n-->\n\n<!DOCTYPE html>\n<html>\n<head>\n  <style>\n  body {\n      background-color: lightblue;\n  }\n\n  h1 {\n      color: white;\n      text-align: center;\n  }\n\n  p {\n      font-family: verdana;\n      font-size: 20px;\n  }\n  </style>\n</head>\n<body>\n\n<h1>Heading</h1>\n<p>This is a paragraph.</p>\n\n</body>\n</html>\n\n");

  slide.code("IDs and classes", null, "\n<!-- The concept if id and class remains the same -->\n<style>\n#para1 {\n    text-align: center;\n    color: red;\n}\n.center {\n    text-align: center;\n    color: red;\n}\n</style>\n\n<p id=\"para1\">Hello World!</p>\n\n<p>This paragraph is not \n    affected by the style.</p>\n<h1 class=\"center\">Red and \n    center-aligned heading</h1>\n<p class=\"center\">Red and \n    center-aligned paragraph.</p>");

  slide.code("Styling Properties", null, "<style>\n  h2 { background-color:#FF0000;\n       color: yellow }\n\n  div.div1{\n\n    border: 1px solid black;\n    margin-top: 100px;\n    margin-bottom: 100px;\n    margin-right: 150px;\n    margin-left: 80px;\n  }\n\n  div.div2{\n    border: 1px solid black;\n    background-color: rgb(255, 255, 0);\n    padding: 50px 30px 50px 80px;\n  }\n\n  div.div3 {\n    height: 200px;\n    width: 50%;\n    background-color: powderblue;\n  }\n</style>\n\n<!-- Colors -->\n<h2>  \nBackground and Text Color\n</h2>\n\n<!-- margins -->\n<div class=\"div1\">div element </div>\n\n<!-- padding -->\n<div class=\"div2\">Another Div</div>\n\n<!-- Height & Width -->\n<div class=\"div3\"> Just Another Div </div>\n");

  slide.code("More Properties", null, "<!-- Borders -->\n<style>\n  p.one {\n    border-style: solid;\n    border-width: 5px;\n}\np.two {\n    border-style:  dashed solid;\n    border-width: medium;\n}\np.three {\n    border-style: dotted;\n    border-width: 2px;\n}\np.four {\n    border-style: ridge;\n    border-width: thick;\n}\np.five {\n    border-style: inset;\n    border-width: 15px;\n}\np.six {\n    border-style: outset;\n    border-width: thick;\n}\np.seven {\n    border-style: solid;\n    border-width: 2px 10px 4px 20px;\n}\ndiv {\n    background-color: lightgrey;\n    width: 300px;\n    border: 15px solid green;\n    padding: 25px;\n    margin: 25px;\n}\n</style>\n \n<p class=\"one\">Some text.</p>\n<p class=\"two\">Some text.</p>\n<p class=\"three\">Some text.</p>\n<p class=\"four\">Some text.</p>\n<p class=\"five\">Some text.</p>\n<p class=\"six\">Some text.</p>\n<p class=\"seven\">Some text.</p>\n<div> The CSS box model is essentially a \nbox that wraps around every HTML element. \nIt consists of: borders, padding, margins,\n and the actual content. </div>");

  slide.code("Text Properties", null, "\n<style>\n\np.uppercase { text-transform: uppercase; }\np.lowercase { text-transform: lowercase; }\np.capitalize { text-transform: capitalize; }\n\np.lowercase { text-indent: 50px; }\np.shadow { background-color:yellow;\n           text-shadow: 3px 2px red;}\n\n</style>\n<!-- Case -->\n<p class=\"uppercase\">This is some text.</p>\n<p class=\"lowercase\">This is some text.</p>\n<p class=\"capitalize\">This is some text.</p>\n\n<!-- Shadow -->\n<p class=\"shadow\"> This is some text </p>\n");

  slide.code("Styling on events", null, "<!-- We can apply different styles on each event -->\n\n<style>\n/* unvisited link */\nbutton {\n    background-color: yellow;\n    color: black;\n    padding: 14px 25px;\n    text-align: center;\n}\nbutton:hover {\n  background-color:lime;\n}\n\nbutton:active {\n  background-color:cyan;\n}\n\n</style>\n\n<button> click here </button>\n");

  slide.code("Combinators", null, "<!-- \nThere are four different combinators in CSS3:\ndescendant selector (space)\nchild selector (>)\nadjacent sibling selector (+)\ngeneral sibling selector (~)\n-->\n<style>\ndiv > p {\n    background-color: yellow;\n}\ndiv   p {\n    color: blue;\n}\ndiv + p {\n    background-color: cyan;\n}\ndiv ~ p {\n    color: darkgreen;\n}\n</style>\n<div>\n  <p>Paragraph 1 in the div.</p>\n  <p>Paragraph 2 in the div.</p>\n\n  <span><p>Paragraph 3 in the div.</p></span> \n  <!-- not Child but Descendant -->\n</div>\n<p>Paragraph 4. Not in a div.</p>\n<p>Paragraph 5. Not in a div.</p>");

  slide.code("Tables", null, "<style>\n#table1, #table1 td, #table1 th {    \n    border: 1px solid #ddd;\n    text-align: left;\n}\n\n#table1 {\n    border-collapse: collapse;\n    width: 100%;\n}\n\n#table1  th, #table1 td {\n    padding: 15px;\n}\n\n#table2 , #table2 th, #table2 td {\n\n    border: 1px solid black;\n}\n</style>\n<table id=\"table1\">\n  <tr>\n    <th>Firstname</th> <th>Lastname</th> <th>Savings</th> </tr>\n  <tr>\n    <td>Peter</td> <td>Griffin</td> <td>$100</td> </tr>\n</table>\n<table id=\"table2\">\n  <tr>\n    <th>Firstname</th> <th>Lastname</th> <th>Savings</th> \n  </tr> \n  <tr>\n    <td>Peter</td> <td>Griffin</td> <td>$100</td> \n  </tr>\n  \n</table>");

  slide.code("Images", null, "<style>\nimg {\n    opacity: 0.5;\n}\n\nimg:hover {\n    opacity: 1.0;\n}\n\ndiv.img {\n    margin: 5px;\n    border: 1px solid #ccc;\n    float: left;\n    width: 180px;\n}\n\ndiv.img:hover {\n    border: 1px solid #777;\n}\n\ndiv.img img {\n    width: 100%;\n    height: auto;\n}\n\ndiv.desc {\n    padding: 15px;\n    text-align: center;\n}\n\n</style>\n<div>\n<img src=\"data/logo_UCSD.png\" \n    width=\"170\" height=\"100\">\n</div>\n\n<div class=\"img\">\n  <a target=\"_blank\" href=\"data/img_forest.jpg\">\n    <img src=\"data/img_forest.jpg\" width=\"300\" height=\"200\">\n  </a>\n  <div class=\"desc\">Add a description of the image here</div>\n</div>\n\n<div class=\"img\">\n  <a target=\"_blank\" href=\"data/img_forest.jpg\">\n    <img src=\"data/img_forest.jpg\" width=\"600\" height=\"400\">\n  </a>\n  <div class=\"desc\">Add a description of the image here</div>\n</div>\n\n<div class=\"img\">\n  <a target=\"_blank\" href=\"data/img_lights.jpg\">\n    <img src=\"data/img_lights.jpg\" width=\"600\" height=\"400\">\n  </a>\n  <div class=\"desc\">Add a description of the image here</div>\n</div>");

  slide.title("Some CSS3");

  slide.code(" More styling", null, "<style>\n#rcorners1 {\n    border-radius: 25px;\n    background: #73AD21;\n    padding: 20px; \n    width: 200px;\n    height: 150px;    \n}\n#bImage {\n    background-image: url(data/img_forest.jpg);\n    padding: 15px;\n    color:white;\n}\n</style>\n\n<!-- Rounder Corners -->\n<p id=\"rcorners1\">Rounded corners!</p>\n\n<!-- Background Images -->\n<div id=\"bImage\">\n  <h1>Heading</h1>\n  <p>Some text</p>\n</div>");

  slide.code("A bit more", null, "<style>\nh1 {\n    text-shadow: 2px 2px 5px red;\n}\nh2 {\n    color: white;\n    text-shadow: 1px 1px 2px black, 0 0 25px blue, 0 0 5px darkblue;\n}\ndiv.polaroid {\n  width: 250px;\n  box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);\n  text-align: center;\n}\n</style>\n\n<h1>Text-shadow effect!</h1>\n<h2> Another Text-shadow effect!</h2>\n\n<!-- Image shadow -->\n<div class=\"polaroid\">\n  <img src=\"data/img_forest.jpg\" style=\"width:100%\">\n  <div class=\"container\">\n    <p>Hardanger, Norway</p>\n  </div>\n</div>\n\n");

  slide.code("Transformations", null, "<style>\n#skewed {\n    -ms-transform: skewX(20deg); /* IE 9 */\n    -webkit-transform: skewX(20deg); /* Safari */\n    transform: skewX(20deg); /* Standard syntax */\n}\n#scaled {\n    margin: 50px 0px 50px 100px;\n    width: 200px;\n    height: 40px;\n    border: 1px solid black;\n    -ms-transform: scale(2,3); /* IE 9 */\n    -webkit-transform: scale(2,3); /* Safari */\n    transform: scale(2,3); /* Standard syntax */\n}\n#rotated {\n    margin-top:150px;\n    -ms-transform: rotate(20deg); /* IE 9 */\n    -webkit-transform: rotate(20deg); /* Safari */\n    transform: rotate(20deg);\n    border: 1px solid black;\n}\n#translated {\n    width: 300px;\n    height: 100px;\n    border: 1px solid black;\n    -ms-transform: translate(50px,100px); /* IE 9 */\n    -webkit-transform: translate(50px,100px); /* Safari */\n    transform: translate(50px,100px); /* Standard syntax */\n}\n</style>\n\n<div id=\"skewed\">\nThis div element is skewed 20 degrees along the X-axis.\n</div>\n<div id=\"scaled\">\nThis div element is scaled 2 times.\n</div>\n<div id =\"rotated\">\nThis div is rotated\n</div>\n<div id=\"translated\">\nThis div is translated\n</div>");

  slide.code("Transitions", null, "<style> \ndiv {\n    width: 100px;\n    height: 100px;\n    background: red;\n    \n}\n\ndiv:hover {\n    width: 300px;\n    height: 300px;\n    -webkit-transition: width 2s, height 4s; /* For Safari 3.1 to 6.0 */\n    transition: width 2s, height 4s;\n}\n</style>\n\n<p><b>Note:</b> This example does not work \nin Internet Explorer 9 and earlier versions.</p>\n\n<div></div>\n\n<p>Hover over the div element above, to see the transition effect.</p>\n");

  slide.title("Most designers prefer frameworks to carry out styling. Check out <a href=\"http://getbootstrap.com/css/\"> Bootstrap CSS ");

  slide.title("Back to HTML");

  slide.code("Layouts", null, "\n<style>\ndiv.container {\n    width: 100%;\n    border: 1px solid gray;\n}\n\nheader, footer {\n    padding: 1em;\n    color: white;\n    background-color: black;\n    clear: left;\n    text-align: center;\n}\n\nnav {\n    float: left;\n    max-width: 160px;\n    margin: 0;\n    padding: 1em;\n}\n\nnav ul {\n    list-style-type: none;\n    padding: 0;\n}\n   \nnav ul a {\n    text-decoration: none;\n}\n\narticle {\n    margin-left: 170px;\n    border-left: 1px solid gray;\n    padding: 1em;\n    overflow: hidden;\n}\n</style>\n\n<div class=\"container\">\n\n<header>\n   <h1>Header</h1>\n</header>\n  \n<nav>\n  Sidebar\n</nav>\n\n<article>\n  <h1>Heading</h1>\n  <p>Text</p>\n</article>\n\n<footer>Footer</footer>\n\n</div>\n");

  slide.code("Canvas", empty_svg, "<canvas id=\"myCanvas\" width=\"200\" height=\"100\" \nstyle=\"border:1px solid #d3d3d3;\">\n</canvas>\n\n<script>\nvar c = document.getElementById(\"myCanvas\");\nvar ctx = c.getContext(\"2d\");\nctx.moveTo(0,0);\nctx.lineTo(200,100);\nctx.stroke();\n</script>\n\n<canvas id=\"myCanvas2\" width=\"200\" height=\"100\" \nstyle=\"border:1px solid #d3d3d3;\"></canvas>\n\n<script>\nvar c = document.getElementById(\"myCanvas2\");\nvar ctx = c.getContext(\"2d\");\nctx.beginPath();\n// x, y, r, startAngle, endAngle \nctx.arc(95,50,40,0,2*Math.PI);\nctx.stroke();\n</script> \n\n<canvas id=\"myCanvas3\" width=\"200\" height=\"100\" \nstyle=\"border:1px solid #d3d3d3;\"></canvas>\n\n<script>\nvar c = document.getElementById(\"myCanvas3\");\nvar ctx = c.getContext(\"2d\");\n// Create gradient\n// xStart, yStart, xEnd, yEnd\nvar grd = ctx.createLinearGradient(0,0,200,0);\ngrd.addColorStop(0,\"red\");\ngrd.addColorStop(1,\"white\");\n// Fill with gradient\nctx.fillStyle = grd;\nctx.fillRect(10,10,150,80);\n</script>\n");

  slide.code("SVG", null, "<svg width=\"100\" height=\"100\">\n  <circle cx=\"50\" cy=\"50\" r=\"40\"\n  stroke=\"green\" stroke-width=\"4\" fill=\"yellow\" />\n<br/>\n\n<svg width=\"200\" height=\"100\">\n  <rect  width=\"200\" height=\"100\"\n  stroke=\"green\" stroke-width=\"4\" fill=\"yellow\" />\n<br/>\n\n<svg width=\"400\" height=\"180\">\n  <rect x=\"50\" y=\"20\" rx=\"20\" ry=\"20\" width=\"150\" height=\"150\"\n  style=\"fill:red;stroke:black;stroke-width:5;opacity:0.5\" />\n</svg>\n<br/>\n\n<svg width=\"300\" height=\"200\">\n  <polygon points=\"100,10 40,198 190,78 10,78 160,198\"\n  style=\"fill:lime;stroke:purple;stroke-width:5;fill-rule:evenodd;\" />\n</svg>");
  
  slide.title('<a href="http://svg-wow.org/camera/camera.xhtml"><h1>SVG: Camera</h1></a>');
  
  slide.title('<a href="http://debeissat.nicolas.free.fr/svg3d.php"><h1>3D SVG</h1></a>');
 
  slide.title('<a href="http://lavadip.com/experiments/jigsaw/"><h1>Jigsaw</h1></a>');
  
  slide.title('Further Reading for <a href="https://www.w3schools.com/html/">HTML</a> and <a href="https://www.w3schools.com/css/default.asp">CSS</a></li>');

}).call(this);
