let assetlist = "/api/api_assets";
let vol_1h = "/api/volume1h";
let vol_1d = "/api/volume1d";
let vol_1m = "/api/volume1m";

function initdropdown() {

    var dropdown = d3.select("#selDataset");

    d3.json(assetlist).then((d) => {
        d.forEach(function (id, index) {
            dropdown.append("option").text(id.asset_id).property("value", id.asset_id);
        });
    });
};

initdropdown();

function init() { //default graph to Bitcoin
    /**
     * JSON output data for reference
     * "time_period_start": "2017-08-09T14:31:00.0000000Z",
     * "time_period_end": "2017-08-09T14:32:00.0000000Z",
     * "time_open": "2017-08-09T14:31:01.0000000Z",
     * "time_close": "2017-08-09T14:31:46.0000000Z",
     * "price_open": 3255.590000000,
     * "price_high": 3255.590000000,
     * "price_low": 3244.740000000,
     * "price_close": 3244.740000000,
     * "volume_traded": 16.903274550,
     * "trades_count": 31 * 
     */

    d3.json(url).then(function (data) {

        var startDate = [];
        var endDate = [];
        var dates = [];
        var openingPrices = [];
        var highPrices = [];
        var lowPrices = [];
        var closingPrices = [];
        var stock = "BTC"

        data.forEach(function (d) {
            startDate.push(d.time_period_start);
            endDate.push(d.time_period_end);
            dates.push(d.time_open);
            openingPrices.push(d.price_open);
            highPrices.push(d.price_high);
            lowPrices.push(d.price_low);
            closingPrices.push(d.price_close);
        });

        var trace1 = {
            type: "scatter",
            mode: "lines",
            //name: name,
            x: dates,
            y: closingPrices,
            line: {
                color: "#063970"
            }
        };

        // Candlestick Trace
        var trace2 = {
            type: "candlestick",
            x: dates,
            high: highPrices,
            low: lowPrices,
            open: openingPrices,
            close: closingPrices
        };

        var data = [trace1, trace2];
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
                type: "date"
            },
            yaxis: {
                autorange: true,
                type: "linear"
            },
            showlegend: false
        };

        Plotly.newPlot("mainchartdiv", data, layout);
    });
} //END DEFAULT PLOT

init();

// Call updatePlotly() when a change takes place to the DOM
d3.select("#selDataset").on("change", updatePlotly);

// This function is called when a dropdown menu item is selected
function updatePlotly() {
    // Use D3 to select the dropdown menu
    var dropdownMenu = d3.select("#selDataset");
    // Assign the value of the dropdown menu option to a variable
    var dataset = dropdownMenu.property("value");
    //console.log(dataset);

    var url = `https://rest.coinapi.io/v1/ohlcv/` + dataset + `/USD/latest?period_id=1DAY&apikey=${api_key}`;
    console.log(url);

    d3.json(url).then(function (data) {

        var startDate = [];
        var endDate = [];
        var dates = [];
        var openingPrices = [];
        var highPrices = [];
        var lowPrices = [];
        var closingPrices = [];
        var stock = dataset;

        data.forEach(function (d) {
            startDate.push(d.time_period_start);
            endDate.push(d.time_period_end);
            dates.push(d.time_open);
            openingPrices.push(d.price_open);
            highPrices.push(d.price_high);
            lowPrices.push(d.price_low);
            closingPrices.push(d.price_close);
        });

        var trace1 = {
            type: "scatter",
            mode: "lines",
            //name: name,
            x: dates,
            y: closingPrices,
            line: {
                color: "#063970"
            }
        };

        // Candlestick Trace
        var trace2 = {
            type: "candlestick",
            x: dates,
            high: highPrices,
            low: lowPrices,
            open: openingPrices,
            close: closingPrices
        };

        var data = [trace1, trace2];
        //console.log(data)

        var layout = {
            title: `${stock} closing prices in $USD`,
            font: {
                family: 'ABeeZee',
                size: 12,
                color: '#7f7f7f'
              },
            xaxis: {
                range: [startDate, endDate],
                type: "date"
            },
            yaxis: {
                autorange: true,
                type: "linear"
            },
            showlegend: false
        };

        Plotly.newPlot("mainchartdiv", data, layout);
    })
}