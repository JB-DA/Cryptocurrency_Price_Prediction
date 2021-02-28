let assetlist = "/api/api_assets";

function initdropdown() {

    var dropdown = d3.select("#selDataset");

    d3.json(assetlist).then((d) => {
        d.forEach(function (id, index) {
            dropdown.append("option").text(id.asset_id).property("value", id.asset_id);
        });
    });
};

initdropdown();