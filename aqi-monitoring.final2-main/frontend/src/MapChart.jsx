import React, { useEffect } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const stateCoordinates = {
  "Maharashtra": [19.7515, 75.7139],
  "Chhattisgarh": [21.2787, 81.8661],
  "Delhi": [28.7041, 77.1025],
  "Karnataka": [15.3173, 75.7139],
  "Uttar Pradesh": [26.8467, 80.9462],
  "Tamil Nadu": [11.1271, 78.6569],
  "West Bengal": [22.9868, 87.8550],
  "Gujarat": [22.2587, 71.1924],
  "Telangana": [17.1231, 79.2088],
  "Rajasthan": [27.0238, 74.2179],
  "India": [20.5937, 78.9629]
};

const districtCoordinates = {
  "Mumbai": [19.0760, 72.8777],
  "Pune": [18.5204, 73.8567],
  "Nagpur": [21.1458, 79.0882],
  "Raipur": [21.2514, 81.6296],
  "Bhilai": [21.1938, 81.3509],
  "Durg": [21.1904, 81.2849],
  "Bilaspur": [22.0797, 82.1409],
  "New Delhi": [28.6139, 77.2090],
  "North Delhi": [28.6833, 77.1667],
  "South Delhi": [28.5355, 77.2410],
  "Bengaluru": [12.9716, 77.5946],
  "Mysuru": [12.2958, 76.6394],
  "Mangaluru": [12.9141, 74.8560],
  "Lucknow": [26.8467, 80.9462],
  "Kanpur": [26.4499, 80.3319],
  "Noida": [28.5355, 77.3910],
  "Chennai": [13.0827, 80.2707],
  "Coimbatore": [11.0168, 76.9558],
  "Madurai": [9.9252, 78.1198],
  "Kolkata": [22.5726, 88.3639],
  "Howrah": [22.5958, 88.3110],
  "Darjeeling": [27.0360, 88.2636],
  "Ahmedabad": [23.0225, 72.5714],
  "Surat": [21.1702, 72.8311],
  "Vadodara": [22.3072, 73.1812],
  "Hyderabad": [17.3850, 78.4867],
  "Warangal": [18.0000, 79.5833],
  "Jaipur": [26.9124, 75.7873],
  "Jodhpur": [26.2389, 73.0243]
};
const indiaBounds = [
  [6.7535, 68.1623],
  [37.1004, 97.3956]
];

function MapUpdater({ selectedState, selectedDistrict }) {
  const map = useMap();
  
  useEffect(() => {
    let coords = stateCoordinates["India"];
    let zoomLevel = 4.5;

    if (selectedDistrict && districtCoordinates[selectedDistrict]) {
      coords = districtCoordinates[selectedDistrict];
      zoomLevel = 10;
    } else if (selectedState && stateCoordinates[selectedState]) {
      coords = stateCoordinates[selectedState];
      zoomLevel = 6;
    }

    map.flyTo(coords, zoomLevel, { animate: true, duration: 1.5 });
  }, [selectedState, selectedDistrict, map]);

  return null;
}

const MapChart = ({ selectedState, selectedDistrict }) => {
  return (
    <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}>
      <MapContainer 
        center={[20.5937, 78.9629]} 
        zoom={4.5} 
        minZoom={4.5}
        maxBounds={indiaBounds}
        maxBoundsViscosity={1.0}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
        />
        <MapUpdater selectedState={selectedState} selectedDistrict={selectedDistrict} />
      </MapContainer>
    </div>
  );
};

export default MapChart;
