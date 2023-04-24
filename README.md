# Stay-Alive-Think-and-Drive-App

## Description

The Stay Alive Think and Drive App is a live app that can help you plan your journey by providing intelligent information about your planned travel route.

You can type in any set of destination and source locations (within Georgia) to get a route along with color-coded segments according to the danger level (based on past accident data),
odds of accident danger according to current weather conditions and statistics.


## Installation

A guide on how to install and set up the Stay Alive Think and Drive App

0. Make sure you have `npm` installed on your system ([Reference here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm))
1. Clone the repository on your local machine using `git clone`
2. Make the `backend-exec` and `frontend-exec` files executable by using `chmod +x backend-exec` and `chmod +x frontend-exec` respectively 
3. Run the `backend-exec` executable on the terminal using the command `./backend-exec`
4. Open a new terminal, and run the `frontend-exec` executable on the terminal using the command `./frontend-exec`
5. Go to the address http://127.0.0.1:5000/ on your browser. You should be able to see the app interface on the above link.


## Execution

- You can enter any destination and source locations (within Georgia) into the form and click the submit button to get the results. 

- The current latency to get the results might be high (depending on your system's resources), but this issue will be resolved with the AWS-deployed version of the app in the future.

**If you want to run the code locally, please contact me @ [Siddharth Singh Solanki](mailto:siddharth.solanki@gatech.edu) for the API KEY.**

After receiving the API KEY run the executable `secret-key` and paste the API KEY onto the terminal. This will create the .env files necessary for the execution of the code.