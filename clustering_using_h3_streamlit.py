import streamlit as st
import pandas as pd
import h3
import folium
from datetime import date
from collections import defaultdict, deque
from streamlit_folium import folium_static
import numpy as np
import colorsys


# Cache data loading
@st.cache_resource
def get_data():
    df = pd.read_csv('data/processed_data.csv')
    return df

@st.cache_data
def get_min_max_date():
    min_date = pd.to_datetime(get_data()["date"]).min().date()
    max_date = pd.to_datetime(get_data()["date"]).max().date()
    return min_date, max_date

# Transport mode configurations
TRANSPORT_CONFIGS = {
    'walking': {'resolution': 9, 'merge_distance': 2},  # ~1km radius
    'bike': {'resolution': 8, 'merge_distance': 4},     # ~2-3km radius
    'van': {'resolution': 7, 'merge_distance': 6},      # ~5-6km radius
    'truck': {'resolution': 6, 'merge_distance': 5}     # ~10km radius
}

# Calculate area for H3 resolution
def get_resolution_area(resolution):
    # Get area of a hexagon at this resolution
    hex_area = h3.cell_area(h3.latlng_to_cell(0, 0, resolution), unit='km^2')
    radius = np.sqrt(hex_area / (2.598076)) # Convert hex area to radius
    return hex_area, radius

class DensityH3Clustering:
    def __init__(self, transport_mode='van', min_density=5):
        config = TRANSPORT_CONFIGS.get(transport_mode)
        self.resolution = config['resolution']
        self.merge_distance = config['merge_distance']
        self.min_density = min_density
        self.transport_mode = transport_mode
        
        # Calculate coverage areas
        self.hex_area, self.hex_radius = get_resolution_area(self.resolution)
        self.coverage_radius = self.hex_radius * self.merge_distance
        
    def get_cell_densities(self, df):
        """Count orders per H3 cell"""
        cell_counts = defaultdict(int)
        for _, row in df.iterrows():
            h3_index = h3.latlng_to_cell(row['lat'], row['lng'], self.resolution)
            cell_counts[h3_index] += 1
        return cell_counts
    
    def merge_adjacent_clusters(self, all_cells, cell_counts):
        """
        Group cells into clusters based on spatial proximity and density patterns
        
        Parameters:
        -----------
        all_cells: set
            Set of all H3 cell indices
        cell_counts: dict
            Dictionary mapping H3 indices to order counts
        """
        clusters = []
        visited = set()

        # Sort cells by density for processing high-density cores first
        sorted_cells = sorted(all_cells, key=lambda x: cell_counts.get(x, 0), reverse=True)
        
        def bfs_cluster(start_cell):
            cluster = set()
            queue = deque([start_cell])
            
            while queue:
                cell = queue.popleft()
                if cell in visited:
                    continue
                    
                visited.add(cell)
                cluster.add(cell)
                
                # Get neighboring cells
                neighbors = h3.grid_disk(cell, self.merge_distance)
                for neighbor in neighbors:
                    if neighbor in all_cells and neighbor not in visited:
                        # Check if neighbor has sufficient density or is adjacent to high-density cell
                        neighbor_density = cell_counts.get(neighbor, 0)
                        current_density = cell_counts.get(cell, 0)
                        
                        if (neighbor_density >= self.min_density or 
                            current_density >= self.min_density):
                            queue.appendleft(neighbor)  # Priority for high-density
                        else:
                            # Add low-density cells if they form a bridge between dense areas
                            dense_neighbors = any(
                                cell_counts.get(n, 0) >= self.min_density 
                                for n in h3.grid_disk(neighbor, 1)
                            )
                            if dense_neighbors:
                                queue.append(neighbor)
                
            return cluster

        # Process cells to form clusters
        for cell in sorted_cells:
            if cell not in visited and cell_counts.get(cell, 0) >= self.min_density:
                cluster = bfs_cluster(cell)
                if cluster:  # Only add if cluster was formed
                    clusters.append(cluster)

        return clusters


    def generate_distinct_colors(self, n):
        """Generate n distinct colors"""
        hsv_colors = [(i/n, 0.8, 0.9) for i in range(n)]
        return ['#%02x%02x%02x' % tuple(int(x*255) for x in colorsys.hsv_to_rgb(*hsv)) 
                for hsv in hsv_colors]

    def visualize_clusters(self, df, output_file='density_clusters.html'):
        """
        Create interactive map showing:
        - Clustered points
        - H3 hexagon boundaries
        - Different colors for each cluster
        """
        # Check for valid data
        if df.empty:
            raise ValueError("No data to visualize")
            
        # Safely calculate center
        valid_lats = df['lat'].dropna()
        valid_lngs = df['lng'].dropna()
        
        if valid_lats.empty or valid_lngs.empty:
            raise ValueError("No valid coordinates found in data")
            
        center_lat = valid_lats.mean()
        center_lng = valid_lngs.mean()
        m = folium.Map(location=[center_lat, center_lng], zoom_start=11)
        
        # Generate distinct colors for clusters
        n_clusters = len(df['cluster'].unique())
        colors = self.generate_distinct_colors(n_clusters)
        
        # Draw hexagons and points for each cluster
        for cluster_id in df['cluster'].unique():
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = df[df['cluster'] == cluster_id]
            color = colors[cluster_id]
            
            # Get unique H3 cells in cluster
            h3_cells = set(cluster_data['h3_index'])
            
            # Draw hexagon boundaries
            for cell in h3_cells:
                boundaries = h3.cell_to_boundary(cell)
                folium.Polygon(
                    locations=boundaries,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_opacity=0.2
                ).add_to(m)
            
            # Draw points
            for _, point in cluster_data.iterrows():
                folium.CircleMarker(
                    location=[point['lat'], point['lng']],
                    radius=6,
                    color=color,
                    fill=True,
                    popup=f"Cluster: {cluster_id}<br>Orders: {len(cluster_data)}"
                ).add_to(m)
        
        # Add transport mode info to map
        folium.Rectangle(
            bounds=[[center_lat-0.1, center_lng-0.2], [center_lat-0.05, center_lng]],
            color="black",
            fill=True,
            popup=f"Transport Mode: {self.transport_mode}<br>Resolution: {self.resolution}"
        ).add_to(m)
                
        m.save(output_file)

    def fit_predict(self, df):
        """
        Perform clustering on filtered data focusing on spatial/density patterns
        """
        # Convert to H3
        h3_indexes = []
        for idx, row in df.iterrows():
            h3_idx = h3.latlng_to_cell(row.lat, row.lng, self.resolution)
            h3_indexes.append(h3_idx)
        df['h3_index'] = h3_indexes

        # Get cell densities
        cell_counts = self.get_cell_densities(df)
        all_cells = set(h3_indexes)
        
        # Merge cells into clusters
        clusters = self.merge_adjacent_clusters(all_cells, cell_counts)
        
        # Assign cluster labels and add density information
        df['cluster'] = -1
        df['cell_density'] = df['h3_index'].map(cell_counts)
        
        for cluster_id, cluster_cells in enumerate(clusters):
            mask = df['h3_index'].isin(cluster_cells)
            df.loc[mask, 'cluster'] = cluster_id
            
            # Get unique cells in cluster
            cluster_cells_data = {
                cell: cell_counts[cell] 
                for cell in cluster_cells
            }
            
            print(f"\nCluster {cluster_id} Statistics:")
            print(f"Total orders: {sum(cluster_cells_data.values())}")
            print(f"High density cells: {sum(1 for count in cluster_cells_data.values() if count >= self.min_density)}")
            print(f"Low density cells: {sum(1 for count in cluster_cells_data.values() if count < self.min_density)}")
            print(f"Average density: {sum(cluster_cells_data.values()) / len(cluster_cells_data):.2f}")
        
        return df


def create_distinct_colors(n):
    """Create n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i/n
        # Bright, saturated colors
        colors.append(f'#{int(colorsys.hsv_to_rgb(hue, 0.8, 0.9)[0]*255):02x}'
                     f'{int(colorsys.hsv_to_rgb(hue, 0.8, 0.9)[1]*255):02x}'
                     f'{int(colorsys.hsv_to_rgb(hue, 0.8, 0.9)[2]*255):02x}')
    return colors


def main():
    st.title("Package Delivery Clustering Analysis")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date filter
    min_date, max_date = get_min_max_date()
    selected_date = st.sidebar.date_input(
        "Select Date",
        min_value=min_date,
        max_value=max_date,
        value=min_date
    )
    
    # Load data
    df = get_data()
    
    # County filter
    counties = sorted(df['county'].unique())
    selected_county = st.sidebar.selectbox(
        "Select County",
        counties
    )
    
    # Transport mode filter
    transport_mode = st.sidebar.selectbox(
        "Select Transport Mode",
        list(TRANSPORT_CONFIGS.keys())
    )
    
    # Show resolution and coverage information
    resolution = TRANSPORT_CONFIGS[transport_mode]['resolution']
    merge_distance = TRANSPORT_CONFIGS[transport_mode]['merge_distance']
    hex_area, hex_radius = get_resolution_area(resolution)
    coverage_radius = hex_radius * merge_distance
    
    st.sidebar.info(f"""
    Resolution Level: {resolution}
    Cell Area: {hex_area:.2f} km²
    Coverage Radius: {coverage_radius:.2f} km
    """)
    
    # Filter data
    filtered_df = df[
        (pd.to_datetime(df["date"]).dt.date == selected_date) &
        (df["county"] == selected_county)
    ]
    
    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return
    
    # Perform clustering
    clusterer = DensityH3Clustering(transport_mode=transport_mode)
    clustered_df = clusterer.fit_predict(filtered_df)
    
    # Create map
    center_lat = filtered_df['lat'].mean()
    center_lng = filtered_df['lng'].mean()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=11)

    # Generate distinct colors for clusters
    n_clusters = len(set(clustered_df['cluster'].unique()) - {-1})
    colors = create_distinct_colors(n_clusters)

    # Store cluster data for interaction
    cluster_data = {}

    # Add clusters to map
    for cluster_id in clustered_df['cluster'].unique():
        if cluster_id == -1:
            continue
            
        cluster_points = clustered_df[clustered_df['cluster'] == cluster_id]
        cluster_cells = set(cluster_points['h3_index'])
        
        # Calculate cluster area
        cluster_area = len(cluster_cells) * hex_area
        
        color = colors[cluster_id]
        
        # Store cluster info
        cluster_data[cluster_id] = {
            'points': cluster_points,
            'area': cluster_area,
            'orders': len(cluster_points)
        }
        
        # Draw hexagons for cluster
        for cell in cluster_cells:
            boundaries = h3.cell_to_boundary(cell)
            folium.Polygon(
                locations=boundaries,
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.2,
                popup=f"""
                Cluster: {cluster_id}
                Area: {cluster_area:.2f} km²
                Orders: {len(cluster_points)}
                """
            ).add_to(m)
        
        # Add points for orders in cluster
        for _, point in cluster_points.iterrows():
            folium.CircleMarker(
                location=[point['lat'], point['lng']],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Order in Cluster {cluster_id} = {len(cluster_points)}"
            ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; 
        top: 50px; right: 50px; z-index: 1000;
        padding: 10px; background-color: white;
        border: 2px solid grey; border-radius: 5px">
    '''
    for cluster_id in cluster_data.keys():
        color = colors[cluster_id]
        legend_html += f'''
        <div>
            <span style="background-color: {color}; 
                padding: 0 10px; margin-right: 5px">
            </span>
            Cluster {cluster_id}
        </div>
        '''
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display map
    st.subheader("Clustering Results")
    folium_static(m)
    
    # Cluster selection
    selected_cluster = st.selectbox(
        "Select Cluster to View Details",
        options=[f"Cluster {cid}" for cid in cluster_data.keys()]
    )
    
    if selected_cluster:
        cluster_id = int(selected_cluster.split()[-1])
        cluster_info = cluster_data[cluster_id]
        
        st.subheader(f"Cluster {cluster_id} Details")
        st.write(f"Area: {cluster_info['area']:.2f} km²")
        st.write(f"Total Orders: {cluster_info['orders']}")
        
        # Display cluster data
        st.dataframe(cluster_info['points'])

if __name__ == "__main__":
    main()