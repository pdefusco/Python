/*
 //Loading Asynchronously:
 var jsondata;
 function loadData() {console.log(jsondata);}
 d3.json('data/olympics.json', function(arg){ jsondata= arg; loadData();})
 
 //Extracting variables needed:
 var data = jsondata.map(function(d){return {year : d.Year, medal : d.Medal, country : d.NOC}});
 */
//window.onload=function(){

// Setting margins
var margin = {top: 20, right: 10, bottom: 100, left:50},
width = 700 - margin.right - margin.left,
height = 500 - margin.top - margin.bottom;

/*------------------------------------------------------------------------------
 define SVG
 Still confused about SVG? see Chapter 3.
 The "g" element is used as a container for grouping objects. The SVG will be
 in "lightgrey" backgorund to help you visualize it.
 See https://developer.mozilla.org/en-US/docs/Web/SVG/Element/g for more info
 ------------------------------------------------------------------------------*/


var svg = d3.select("body")
.append("svg").attr({"width": width + margin.right + margin.left,
                    "height": height + margin.top + margin.bottom
                    });
//.append("g").attr("transform","translate(" + margin.left + "," + margin.right + ")");
//}

/* -----------------------------------------------------------------------------
 SCALE and AXIS are two different methods of D3. See D3 API Refrence for info on
 AXIS and SCALES. See D3 API Refrence to understand the difference between
 Ordinal vs Linear scale.
 ------------------------------------------------------------------------------*/
// define x and y scales
var xScale = d3.scaleBand()
.rangeRound([0,width]).padding(0.2, 0.2);

var yScale = d3.scaleLinear()
.range([height, 0]);

// define x axis and y axis
var xAxis = d3.axisBottom()
.scale(xScale);

var yAxis = d3.axisLeft()
.scale(yScale);

/* -----------------------------------------------------------------------------
 To understand how to import data. See D3 API refrence on CSV. Understand
 the difference between .csv, .tsv and .json files. To import a .tsv or
 .json file use d3.tsv() or d3.json(), respectively.
 ------------------------------------------------------------------------------*/
d3.json("data/olympics.json", function(error,data) {
        if(error) console.log("Error: data not loaded!");
        
        /*----------------------------------------------------------------------------
         Convert data if necessary. We want to make sure our gdp vaulues are
         represented as integers rather than strings. Use "+" before the variable to
         convert a string represenation of a number to an actual number. Sometimes
         the data will be in number format, but when in doubt use "+" to avoid issues.
         ----------------------------------------------------------------------------*/
        data.map(function(d){return
                 {year : d.Year
                 medal : d.Medal
                 country : d.NOC}});
        
        // grouping by year and country
        var entries = d3.nest().key(function(d){return d.year;}).key(function(d){return d.country;}).rollup(function(d) { return d.length; }).entries(data);
        
        // sort by year values
        entries.sort(function(a,b) {
                     return a.key - b.key;
                     });
        
        var final_data = entries.map(function(d){
                                     var datum = {};
                                     
                                     var values = d.values;
                                     
                                     var max_value = d3.max(values, function(d){return d.value;})
                                     
                                     datum.year = d.key;
                                     datum.maxCount = max_value;
                                     datum.countries = values.filter(function(d){
                                                                     if(d.value == max_value){
                                                                     return true;
                                                                     }
                                                                     return false;
                                                                     }).map(function(d){return d.key});
                                     
                                     return datum;
                                     });
        
        // Specify the domains of the x and y scales
        xScale.domain(final_data.map(function(d) { return d.country; }) );
        yScale.domain([0, d3.max(final_data, function(d) { return d.gdp; } ) ]);
        
        svg.selectAll('rect')
        .data(final_data)
        .enter()
        .append('rect')
        .attrs("height", 0)
        .attrs("y", height)
        .transition().duration(3000)
        .delay( function(d,i) { return i * 200; })
        // attributes can be also combined under one .attr
        .attrs({
              "x": function(d) { return xScale(d.country); },
              "y": function(d) { return yScale(d.gdp); },
              "width": xScale.rangeBand(),
              "height": function(d) { return  height - yScale(d.gdp); }
              })
        .styles("fill", function(d,i) { return 'rgb(20, 20, ' + ((i * 30) + 100) + ')'});
        
        
        svg.selectAll('text')
        .data(final_data)
        .enter()
        .append('text')
        .text(function(d){
              return d.country;
              })
        .attrs({
              "x": function(d){ return xScale(d.year) +  xScale.rangeBand()/2; },
              "y": function(d){ return yScale(d.maxCount)+ 12; },
              "font-family": 'sans-serif',
              "font-size": '13px',
              "font-weight": 'bold',
              "fill": 'white',
              "text-anchor": 'middle'
              });
        
        // Draw xAxis and position the label
        svg.append("g")
        .attrs("class", "x axis")
        .attrs("transform", "translate(0," + height + ")")
        .call(xAxis)
        .selectAll("text")
        .attrs("dx", "-.8em")
        .attrs("dy", ".25em")
        .attrs("transform", "rotate(-60)" )
        .styles("text-anchor", "end")
        .attrs("font-size", "10px");
        
        
        // Draw yAxis and postion the label
        svg.append("g")
        .attrs("class", "y axis")
        .call(yAxis)
        .append("text")
        .attrs("transform", "rotate(-90)")
        .attrs("x", -height/2)
        .attrs("dy", "-3em")
        .styles("text-anchor", "middle")
        .text("Total Number of Medals");
        });



/////old
/*
 var dat = d3.json("data/olympics.json", createChart);
 
 var sample;
 
 function loadData(jsondata) {
 return jsondata;
 }
 
 var olympics=d3.json(dataPath, loadData);
 
 
 
 
 /////
 function createChart(data){
 sample = data;
 var entries = d3.nest().key(function(d){return d.Year;}).key(function(d){return d.NOC;}).rollup(function(d) { return d.length; }).entries(sample);
 
 var final_data = entries.map(function(d){
 var datum = {};
 
 var values = d.values;
 
 var max_value = d3.max(values, function(d){return d.value;})
 
 datum.Year = d.key;
 datum.maxCount = max_value;
 datum.countries = values.filter(function(d){
 if(d.value == max_value){
 return true;
 }
 return false;
 }).map(function(d){return d.key});
 
 return datum;
 });
 
 var years = []
 var counts = []
 var nations = []
 
 for ( i=0; i<final_data.length; i++ )
 {
 years.push(final_data[i].Year)
 counts.push(final_data[i].maxCount)
 nations.push(final_data[i].countries)
 }
 
 
 console.log(years)
 console.log(counts)
 console.log(nations)
 
 var svg = d3.select("div.output svg");
 
 var data = counts;
 
 //var data = data.map(function(d){return max(d)})
 //NEW LINE TO SORT
 //var data = d3.nest().sortValues(d3.ascending).entries(data);
 
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
 .text(function(d, i) { return String(d) + " " + nations[i] })
 .style("fill", "white")
 .style("text-anchor", "end");
 }
 
 */


