require('dotenv').config();
const cors = require('cors');
var express = require('express');
var app = express();
var fs = require("fs");
const request = require("request");
const util = require("@googlemaps/google-maps-services-js/dist/util");
var data = [];

const api_pre = "https://maps.googleapis.com/maps/api/directions/json"

app.get('/hello', function (req, res) {
    res.send("Hello World")
});
app.use(cors());
app.get('/dir/:src&:dst', function (req, res) {
    // res.setHeader('Access-Control-Allow-Origin', 'http://localhost:5500');
    // res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, PATCH, DELETE');
    var dest = req.params.dst.replace(/\s/g, '');
    var sour = req.params.src.replace(/\s/g, '');
    var api_string = api_pre + "?destination=" + dest + "&origin=" + sour + "&key=" + process.env.API_KEY
    console.log(api_string);
    request(api_string, { json: true }, function (err, resp, body) {
        if (err) { return console.log(err); }
        console.log(body.status);
        if (body.status == "NOT_FOUND") { return res.send("API Call"); }
        // let rawdata = fs.readFileSync('sid.json');
        // let body = JSON.parse(rawdata);
        
        
        var all_coords = []; 
        var bounds = body.routes[0].bounds;
        var center = {};
        center.lat = (bounds.northeast.lat + bounds.southwest.lat) / 2;
        center.lng = (bounds.northeast.lng + bounds.southwest.lng) / 2;
    
    
        var segment_data = body.routes[0].legs[0].steps;
        var segment_coords;
        var distance;
        var points_to_return = [];
        for (let d = 0; d < segment_data.length; d++) {
            segment_coords = util.decodePath(segment_data[d].polyline.points);
            distance = segment_data[d].distance
            distance = +distance.text.split(" ")[0]
            for (let i = 0; i < segment_coords.length; i++) {
                all_coords.push(segment_coords[i]);
            }
            var points_per_100_metre = Math.floor(0.1 / (distance * 1.6 / all_coords.length));
            for (let b = 0; b < all_coords.length; b++) {
                if (b % points_per_100_metre == 0) {
                    points_to_return.push([all_coords[b], 1])
                }
                else {
                    points_to_return.push([all_coords[b], 0])
                }
            }
            all_coords = [];
        }
        let test = [3, 4];
    
        console.log("Here", test[0]);
        fs.writeFile("./python_code/data.json", JSON.stringify(points_to_return), (err) => {
            if (err) {
                return console.log(err);
            }
            console.log("hereeee")
            var spawn = require("child_process").spawn;
            var process = spawn('python3', ["./py_backend.py"]);
            process.stdout.on('data', async function (data1) {
                var data_show = [];
                var raw = fs.readFileSync('outputs/backend_output.json');
                let acc_prop_output = JSON.parse(raw);
                data_show.push(acc_prop_output);
                
                raw = fs.readFileSync('outputs/hour_acc_hist.json');
                let hourly_acc_hist_data = JSON.parse(raw);
                data_show.push(hourly_acc_hist_data);

                raw = fs.readFileSync('outputs/logOdds_output.json');
                let logOdds_output = JSON.parse(raw);
                data_show.push(logOdds_output);

                res.send(data_show);
    
            });
            process.on('exit', code => {
                if (code != 0) {
                    process.stderr.on('data', async data => { console.log("Error occured in python code") });
                }
            });
    
    
        });
    });
    
});

var server = app.listen(5000, function() {
    var host = server.address().address
    var port = server.address().port
    console.log("Example app listening at http://%s:%s", host, port)
});
