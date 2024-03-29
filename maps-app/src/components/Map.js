import React from 'react'
import { Loader } from "@googlemaps/js-api-loader"

function Map(props) {
  // console.log("bohooo");

  console.log("Inside maps component",props.startL,props.endL)
  const loader = new Loader({
    apiKey: "AIzaSyCU-XID7IaVFN6Skviaf7g0vpUQcg9GdQ8",
    version: "weekly",
  });

  loader.load().then(async () => {
    const google = window.google;
    const { Map } = await google.maps.importLibrary("maps");

    // new Map(document.getElementById("map"), {
    //   center: { lat: -34.397, lng: 150.644 },
    //   zoom: 8,
    // });
    console.log("before_fetch");
    

    // 'https://maps.googleapis.com/maps/api/directions/json?destination=Montreal&origin=Toronto&key=AIzaSyCU-XID7IaVFN6Skviaf7g0vpUQcg9GdQ8'
    fetch("http://127.0.0.1:3000/dir/100midtown&palomawest", {mode:'cors'})
      .then(response => response.json())
      .then(data => {
        console.log("inside_fetch");
        console.log(data)

        var numsegments = 5;
        const map = new Map(document.getElementById("map"), {
          zoom: 16,
          center: data[0][`segment_${3}`]['route'][0],

          mapTypeId: "terrain",
        });
        var color;
        var coord_data;
        for (let i = 1; i <= 5; i++) {
          color = data[0][`segment_${i}`]['api_color'];
         
          console.log(`#${data[0][`segment_${i}`]['api_color']}`, data[0][`segment_${i}`]['route'])
          
          coord_data = data[0][`segment_${i}`]['route'];

          if (i < numsegments) {
            coord_data.push(data[0][`segment_${i + 1}`]['route'][0]);
          }


          new google.maps.Polyline({
            path: coord_data,
            geodesic: true,
            strokeColor: "#" + color,
            strokeOpacity: 1.0,
            strokeWeight: 7.5,
          }).setMap(map);
        }

      });

    console.log("after_fetch");
  });

 

  const mapStyle = {
    backgroundColor: 'green',
    height: '100vh',
    width: '100%',
  };


  return (
    <div id="map" style={mapStyle}>Map</div>
  )
}

export default Map



// return (
//   <div className='h-20 bg-green-400'>Map</div>
// )
// }