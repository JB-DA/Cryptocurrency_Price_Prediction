//alert("The Index Javascript file was loaded...");

var stock = 'BTC'

let chartURL = `/api/assets?asset=${stock}`;
console.log(chartURL)

function init() { //default graph to Bitcoin

    var url = chartURL;
    console.log(url);

    d3.json(url).then(function (data) {

        var startDate = [];
        var endDate = [];
        var dates = [];
        var openingPrices = [];
        var highPrices = [];
        var lowPrices = [];
        var closingPrices = [];
        //var stock = "brentoil"

        data.forEach(function (d) {
            startDate.push(d.time_period_start);
            endDate.push(d.time_period_end);
            dates.push(d.time_open);
            openingPrices.push(d.price_open);
            highPrices.push(d.price_high);
            lowPrices.push(d.price_low);
            closingPrices.push(d.price_close);
        });

        var trace0 = {
            type: "scatter",
            mode: "lines",
            name: "Price",
            x: endDate,
            y: closingPrices,
            line: {
                color: "#063970"
            }
        };

        // Candlestick Trace
        var trace1 = {
            type: "candlestick",
            name: "Movement",
            x: endDate,
            high: highPrices,
            low: lowPrices,
            open: openingPrices,
            close: closingPrices
        };

        var data = [trace0, trace1];
        //console.log(data);

        var layout = {
            title: `${stock} closing prices in $USD`,
            font: {
                family: 'ABeeZee',
                size: 12,
                color: '#7f7f7f'
              },
            xaxis: {
                range: [startDate, endDate],
                type: "date",
                hoverformat: '.2f'
            },
            yaxis: {
                autorange: true,
                type: "linear",
                hoverformat: '.2f   '            
            },
            width: "1000",
            height: "500",
            showlegend: true
        };

        Plotly.newPlot("mainchartdiv", data, layout);
    });
} //END DEFAULT PLOT

init();