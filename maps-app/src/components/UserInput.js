import React, {useState } from 'react'

function UserInput(props) {
  // create two state variables and two setter functions to update their values
  const [startingLocation, setStartingLocation] = useState('');
  const [destination, setDestination] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);

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
    SL = startingLocation;

    console.log("Yo", SL);
    props.onSubmit({0:startingLocation, 1:destination, 2:true});
    DN = destination;

    console.log("Yo", DN);


    // call backend functions here - both map function and log-odds function
    // await fetch('/api/my-backend-function', {
    //   method: 'POST',
    //   body: JSON.stringify({ startingLocation, destination}),
    // });

    setIsSubmitted(true);
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

  // use the two state variables as the values of the starting location and destination fields
  // update the state variables only once the submit button is clicked
  return (
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
  );
}

export default UserInput;