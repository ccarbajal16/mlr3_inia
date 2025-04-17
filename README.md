# MLR3 Spatial Prediction Tool

A Shiny application for spatial prediction using the mlr3 machine learning framework in R.

## Overview

This application provides a user-friendly interface for:

- Loading spatial data (rasters and point data)
- Training machine learning models
- Evaluating model performance
- Generating spatial predictions

## Features

- **Data Input**

  - Upload predictor raster files (.tif)
  - Upload response & coordinates data (.csv)
  - Automatic column detection for coordinates
- **Model Training**

  - Random Forest (Ranger) and Gradient Boosting (GBM) algorithms
  - Configurable train/test split
  - Optional hyperparameter tuning
- **Model Evaluation**

  - Performance metrics (RMSE, R², MAE)
  - Predicted vs. observed plots
  - Feature importance visualization
  - Cross-validation results
- **Spatial Prediction**

  - Adjustable resolution
  - Optional masking
  - Downloadable prediction results

## Getting Started

### Prerequisites

- R (>= 4.0.0)
- Required R packages (see below)

### Installation

1. Clone this repository
2. Install required packages:

```r
install.packages(c(
  "shiny", "shinythemes", "terra", "ranger", "gbm", "data.table",
  "mlr3", "mlr3learners", "mlr3spatiotempcv", "mlr3pipelines",
  "mlr3extralearners", "mlr3tuning", "paradox", "sf", "ggplot2",
  "tidyterra", "viridis"
))
```

### Running the Application

Open the project in RStudio and click "Run App" or run:

```r
shiny::runApp()
```

## Usage

1. **Upload Data**

   - Upload predictor raster files (.tif)
   - Upload CSV file with response variable and coordinates
   - Select coordinate columns and response variable
2. **Configure Model**

   - Select model type (Random Forest or GBM)
   - Set training data proportion
   - Enable hyperparameter tuning (optional)
3. **Train Model**

   - Click "Train Model" button
   - View results in the tabs
4. **Generate Prediction**

   - Set raster resolution
   - Apply mask (optional)
   - Click "Generate Prediction" button
   - Download prediction as GeoTIFF

## Documentation

For more detailed information, see the documentation website (when deployed):

- **[Documentation Website](https://yourusername.github.io/mlr3_app/)** - Complete documentation with user guide and technical details

Or view the documentation files directly:

- [Technical Documentation](docs/technical_documentation.md) - Detailed technical information
- [User Guide](docs/user_guide.md) - Instructions for using the application
- [Project Index](docs/project_index.md) - Overview of project structure

The documentation website is built using [Quarto](https://quarto.org/) and deployed to GitHub Pages.

This application is built on the [mlr3 framework](https://mlr3.mlr-org.com/). For comprehensive documentation on mlr3, please refer to the [mlr3book](https://mlr3book.mlr-org.com/), which is the principal reference for understanding the underlying methodology.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
