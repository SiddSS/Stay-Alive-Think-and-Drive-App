import React, { useState, useEffect } from 'react';

function RouteStatistics() {
  const [totalAccidents, setTotalAccidents] = useState(0);

  useEffect(() => {
    // make an API call to the backend to fetch totalAccidents
    // and set it using setTotalAccidents
    const fetchTotalAccidents = async () => {
      const response = await fetch('/api/sum_num_accidents');
      const data = await response.json();
      setTotalAccidents(data.totalAccidents);
    };

    fetchTotalAccidents();
  }, []);

  return (
    <div>
      <h2>Total Historic Accidents on Route</h2>
      <p style={{ fontSize: '36px' }}>N/A (route not selected)</p>
    </div>
  );
}

export default RouteStatistics;

