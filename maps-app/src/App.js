import React, { useState } from 'react';
import Map from './components/Map';
import OddComp from './components/odd_comp';
// import UserInput from './components/UserInput';
import Graphs from './components/graphs';
import Map2 from './components/Map2';
import { Loader } from "@googlemaps/js-api-loader"

function App() {
  // variables
  var odds;

  // create two state variables and two setter functions to update their values
  const [startingLocation, setStartingLocation] = useState('');
  const [destination, setDestination] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [apiFetched, setApiFetched] = useState(false);

  // two event handlers - user these to update state variables whenever corresponding input fields change
  const handleStartingLocationChange = (event) => {
    setStartingLocation(event.target.value);
    console.log(event.target.value);
  };

  const handleDestinationChange = (event) => {
    setDestination(event.target.value);
    console.log(setDestination(event.target.value));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    console.log("Yo", startingLocation);
    // props.onSubmit({0:startingLocation, 1:destination, 2:true});
    console.log("Yo", destination);
    setIsSubmitted(true);

    // maps code start ######################################################


    console.log("Inside maps component", startingLocation, destination)
    const loader = new Loader({
      apiKey: process.env.REACT_APP_API_KEY,
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
      fetch(`http://127.0.0.1:5000/dir/${startingLocation}&${destination}`, { mode: 'cors' })
        .then(response => response.json())
        .then(data => {
          console.log("inside_fetch");
          console.log(data)
          odds = data[2];
          setApiFetched(true);
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


    //maps code ends #####################################

  };

  const UserInputStyle = {
    backgroundColor: 'lightgray',
    height: '30vh',
    width: '100%',
    padding: '20px'
  };

  const inputContainerStyle = {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    marginBottom: '20px',
    alignItmes: 'center',
  };

  const buttonStyle = {
    backgroundColor: 'blue',
    color: 'white',
    fontWeight: 'bold',
    padding: '10px',
    borderRadius: '5px',
    border: 'none',
    width: '150px',
  };

  const inputLabelStyle = {
    fontWeight: 'bold',
    marginBottom: '5px'
  };

  //userinput code above

  //map js
  const mapStyle = {
    backgroundColor: 'green',
    height: '100vh',
    width: '100%',
  };
  //


  const titleStyle = {
    fontSize: '48px',
    fontWeight: 'bold',
    textAlign: 'center'
  };

  const subtitleStyle = {
    fontSize: '20px',
    textAlign: 'center',
    padding: '20px'
  };

  const mapcontainerStyle = {
    height: '100vh',
    width: '50%',
    float: 'left',
    padding: '20px'
  };

  const rightColumnStyle = {
    display: 'flex',
    flexDirection: 'column',
    width: '50%'
  }

  const componentStyle = {
    padding: '20px'
  }

  return (
    <div>
      <div style={titleStyle}>Stay Alive: Think and Drive</div>
      <div style={subtitleStyle}>Enter a starting and ending address within the state of Georgia. This tool only displays data for accidents on
        major roads and highways.  </div>
      <div style={{ display: 'flex' }}>
        <div style={mapcontainerStyle}>
          <div id="map" style={mapStyle}>Map</div>
        </div>
        <div style={rightColumnStyle}>

          <div style={UserInputStyle}>
            <form onSubmit={handleSubmit}>
              <div style={inputContainerStyle}>
                <label htmlFor="startingLocation" style={inputLabelStyle}>Starting Point:</label>
                <input type="text" id="startingLocation" value={startingLocation} onChange={handleStartingLocationChange} />
              </div>
              <div style={inputContainerStyle}>
                <label htmlFor="destination" style={inputLabelStyle}>Destination:</label>
                <input type="text" id="destination" value={destination} onChange={handleDestinationChange} />
              </div>
              <button type="submit" style={buttonStyle}>Submit</button>
            </form>
            {isSubmitted && (
              <p>Route submitted!</p>
            )}
          </div>




          {/* <div style={componentStyle}><UserInput onSubmit={getData}/></div> */}
          <div style={componentStyle}><Graphs /></div>
          <div style={componentStyle}><OddComp oddsreturn = {odds} /></div>
        </div>
      </div>





    </div>
  );
}
export default App;