# H3 Clustering for Last Mile Delivery Optimization

This Streamlit application demonstrates the power of H3 spatial indexing for optimizing last-mile delivery routes through intelligent clustering. The demo uses processed data from the Amazon Last Mile Routing Research Challenge to showcase how different modes of transportation affect delivery zone planning.

## Background

### H3 Spatial Index
[H3](https://h3geo.org/) is a hexagonal hierarchical geospatial indexing system developed by Uber. It provides a unique way to divide the world into hexagonal cells at different resolutions, making it particularly useful for:
- Delivery zone planning
- Service area optimization
- Location-based clustering

### Amazon Last Mile Routing Research Challenge
The [Amazon Last Mile Routing Research Challenge](https://aws.amazon.com/marketplace/pp/prodview-rqkdusd3nz3mw#resources) provided real-world routing data to encourage innovation in last-mile delivery optimization. This application uses a processed subset of this data to demonstrate practical applications of H3 clustering.

## Data Description
Original data structure - https://github.com/MIT-CAVE/rc-cli/blob/main/templates/data_structures.md

The application uses a processed CSV file as suitable.

Data processing steps included:
1. Extraction of relevant fields from the original dataset
2. Addition of calculated fields for demonstration purposes

## H3 Clustering Implementation

The application demonstrates clustering using H3 with:
- Different resolution levels based on transportation mode
- Dynamic service area calculation
- Delivery point aggregation

Resolution levels by transport mode:
- Walking: Resolution 9 (coverage radius = 0.35 km radius)
- Bike: Resolution 8 (coverage radius = 1.84km radius)
- Van: Resolution 7 (coverage radius = 7.30km radius)
- Truck: Resolution 6 (coverage radius = 16.08 km radius)


## Installation

```bash
# Clone the repository
git clone https://github.com/kapoorabhish/h3_clustering.git
cd h3-clustering

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Requirements

```text
streamlit==1.24.0
h3==3.7.6
pandas==1.5.3
plotly==5.13.1
numpy==1.24.3
```

## How to Run

```bash
# Navigate to the project directory
cd h3-clustering

# Run the Streamlit app
streamlit run clustering_using_h3_streamlit.py
```

The application will open in your default web browser at `http://localhost:8501`

## Using the Application

1. Select a date from the sidebar
2. Choose a county from the available options
3. Select mode of transportation:
   - Walking
   - Bike
   - Van
   - Truck
4. Observe how clusters dynamically adjust based on your selections
5. View metrics for each cluster including:
   - Number of delivery points
   - Estimated delivery time
   - Coverage area

## Demo

![Demo](H3_Clustering.gif)

## Project Structure

```
h3-clustering-demo/
├── clustering_using_h3_streamlit.py    # Main Streamlit application
├── data/
│   └── processed_data.csv  # Processed delivery data
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Note: This is a demonstration application using processed public data. For production use, please refer to the official H3 documentation and Amazon Last Mile Challenge guidelines.*