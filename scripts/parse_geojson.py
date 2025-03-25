import json
import re
from collections import defaultdict

with open('/Users/dB/Documents/repos/github/utilities-dashboard-v3/geo/world.geojson', 'r') as f:
    data = json.load(f)

# Map of zone patterns to states
state_mappings = {
    # Florida patterns
    'US-FLA-': 'Florida',
    
    # Carolinas/Virginia patterns
    'US-CAR-': 'Carolinas',  # This can include North and South Carolina
    'US-CAR-SC': 'South Carolina',
    'US-CAR-SCEG': 'South Carolina',
    'US-CAR-DUK': 'North Carolina',  # Duke Energy covers NC
    'US-CAR-CPLE': 'North Carolina',  # Carolina Power & Light East
    'US-CAR-CPLW': 'North Carolina',  # Carolina Power & Light West
    
    # Tennessee Valley Authority
    'US-TEN-TVA': 'Tennessee',
    
    # Southeastern zones
    'US-SE-': 'Southeastern US',  # General southeastern US 
    'US-SE-SOCO': 'Georgia',  # Southern Company (Georgia Power)
    
    # Midwestern zones (may include Kentucky)
    'US-MIDW-LGEE': 'Kentucky',  # Louisville Gas & Electric/Kentucky Utilities
    
    # Other state patterns
    'US-MIDW-': 'Midwestern US',  # May cover parts of Kentucky and Tennessee
}

# Map for region to specific states (for composite regions)
region_to_states = {
    'Carolinas': ['North Carolina', 'South Carolina'],
    'Southeastern US': ['Alabama', 'Georgia', 'Mississippi'],
    'Midwestern US': ['Kentucky', 'Tennessee', 'Virginia'],
}

# Group zones by state
zones_by_state = defaultdict(list)
zone_count = 0

for feature in data['features']:
    props = feature.get('properties', {})
    zoneName = props.get('zoneName', '')
    countryName = props.get('countryName', '')
    
    # Only process US zones
    if countryName == 'United States':
        # Check if this zone matches any of our patterns
        state_match = None
        for pattern, state in state_mappings.items():
            if zoneName.startswith(pattern):
                state_match = state
                break
        
        if state_match:
            zone_count += 1
            # Get centroid from coordinates
            try:
                geometry = feature.get('geometry', {})
                if geometry.get('type') == 'MultiPolygon':
                    # Calculate centroid from all polygon points
                    all_points = []
                    for polygon in geometry.get('coordinates', []):
                        for ring in polygon:
                            all_points.extend(ring)
                    
                    if all_points:
                        lon_sum = sum(point[0] for point in all_points)
                        lat_sum = sum(point[1] for point in all_points)
                        count = len(all_points)
                        
                        # If this is a composite region, add to all constituent states
                        if state_match in region_to_states:
                            for constituent_state in region_to_states[state_match]:
                                zones_by_state[constituent_state].append({
                                    'zoneName': zoneName,
                                    'centroid': {
                                        'longitude': lon_sum / count,
                                        'latitude': lat_sum / count,
                                        'altitude': 0  # GeoJSON doesn't typically include altitude
                                    },
                                    'region': state_match
                                })
                        else:
                            zones_by_state[state_match].append({
                                'zoneName': zoneName,
                                'centroid': {
                                    'longitude': lon_sum / count,
                                    'latitude': lat_sum / count,
                                    'altitude': 0  # GeoJSON doesn't typically include altitude
                                }
                            })
            except Exception as e:
                print(f'Error processing {zoneName}: {e}')

# Print the results
print(f"Found {zone_count} zones in the southeastern US states.")

for state, zones in sorted(zones_by_state.items()):
    print(f'\n{state}:')
    for zone in sorted(zones, key=lambda z: z['zoneName']):
        centroid = zone['centroid']
        if 'region' in zone:
            print(f"  {zone['zoneName']} - Lat: {centroid['latitude']:.6f}, Lon: {centroid['longitude']:.6f}, Alt: {centroid['altitude']} (Part of {zone['region']})")
        else:
            print(f"  {zone['zoneName']} - Lat: {centroid['latitude']:.6f}, Lon: {centroid['longitude']:.6f}, Alt: {centroid['altitude']}")

# Also provide a list of all US zone names to help debug state matching
print("\nAll US Zone Names:")
us_zones_list = []
for feature in data['features']:
    props = feature.get('properties', {})
    if props.get('countryName') == 'United States':
        us_zones_list.append(props.get('zoneName', ''))

for zone in sorted(us_zones_list):
    print(f"  {zone}") 