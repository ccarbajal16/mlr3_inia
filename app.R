library(shiny)
options(shiny.maxRequestSize = 100*1024^2)
library(shinythemes)
library(terra)
library(ranger)
library(gbm)
library(data.table)
library(mlr3)
library(mlr3learners)
library(mlr3spatiotempcv) # Keep for potential spatial CV option
library(mlr3pipelines)
library(mlr3extralearners)  # For Cubist model
library(sf) # For potential spatial operations/visualizations
library(ggplot2) # For plotting results
library(mlr3tuning) # Add if tuning is implemented later
library(paradox)    # Add if tuning is implemented later
library(tidyterra)  # Add if advanced raster plotting is needed
library(viridis) # For color scales
library(markdown) # For rendering markdown content

# Define UI
ui <- fluidPage(
  # Apply a custom theme with INIA colors
  theme = shinytheme("cerulean"),
  tags$head(
    tags$style(HTML("
      .navbar-default {
        background-color: #006633 !important; /* INIA green color */
        border-color: #004d26;
      }
      .navbar-default .navbar-brand {
        color: white;
      }
      .navbar-default .navbar-nav > li > a {
        color: white;
      }
      .navbar-default .navbar-nav > .active > a,
      .navbar-default .navbar-nav > .active > a:focus,
      .navbar-default .navbar-nav > .active > a:hover {
        background-color: #004d26;
        color: white;
      }
      .btn-primary {
        background-color: #006633;
        border-color: #004d26;
        color: black; /* Changed text color to black for better visibility */
        font-weight: bold;
      }
      .btn-primary:hover, .btn-primary:focus {
        background-color: #004d26;
        border-color: #003d1f;
        color: black; /* Maintain black text on hover */
      }
      h4 {
        color: #006633;
        font-weight: bold;
      }
      .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
      }
      .logo-img {
        height: 60px;
        margin-right: 15px;
      }
      .title-container {
        flex-grow: 1;
      }
    "))
  ),

  # Custom header with logo and title
  div(class = "logo-container",
      tags$img(src = "INIA_LOGO.png", class = "logo-img", alt = "INIA Logo"),
      div(class = "title-container",
          h2("Machine Learning with R and mlr3", style = "margin-top: 0; color: #006633;"),
          p("Spatial Prediction Tool", style = "font-style: italic;")
      )
  ),

  # Main layout
  sidebarLayout(
    sidebarPanel(
      div(class = "panel panel-default",
          div(class = "panel-heading", style = "background-color: #006633; color: white;",
              h4("1. Upload Data", style = "margin: 0; color: white;")),
          div(class = "panel-body",
              # Input for predictor raster files
              fileInput("raster_files", "Upload Predictor Rasters (.tif)",
                        multiple = TRUE,
                        accept = c(".tif", ".tiff")),

              # Input for CSV data (response and coordinates)
              fileInput("csv_file", "Upload Response & Coordinates (.csv)",
                        multiple = FALSE,
                        accept = c(".csv")),

              # UI elements to select columns will be added here later
              uiOutput("coord_x_selector"),
              uiOutput("coord_y_selector"),
              uiOutput("response_selector")
          )
      ),

      div(class = "panel panel-default", style = "margin-top: 15px;",
          div(class = "panel-heading", style = "background-color: #006633; color: white;",
              h4("2. Model Configuration", style = "margin: 0; color: white;")),
          div(class = "panel-body",
              # Select model type
              selectInput("model_type", "Select Model:",
                          choices = c("Random Forest (Ranger)" = "ranger",
                                      "Gradient Boosting (GBM)" = "gbm"),
                          selected = "ranger"),

              # Slider for train/test split ratio
              sliderInput("train_split_ratio", "Training Data Proportion:",
                          min = 0.5, max = 0.9, value = 0.7, step = 0.05)
          )
      ),

      div(class = "panel panel-default", style = "margin-top: 15px;",
          div(class = "panel-heading", style = "background-color: #006633; color: white;",
              h4("3. Hyperparameter Tuning (Optional)", style = "margin: 0; color: white;")),
          div(class = "panel-body",
              checkboxInput("enable_tuning", "Enable Hyperparameter Tuning", value = FALSE),

              # Conditional UI for tuning options
              conditionalPanel(
                condition = "input.enable_tuning == true",
                # Only Random Search for simplicity now
                # selectInput("tuning_method", "Tuning Method:", choices = c("Random Search" = "random_search")),
                numericInput("tuning_evals", "Number of Tuning Evaluations:", value = 10, min = 2, max = 100),
                # Fixed inner CV for tuning
                # selectInput("tuning_resampling", "Tuning Resampling:", choices = c("3-Fold CV" = "cv3")),
                helpText("Tuning uses Random Search with 3-fold CV on the training data subset.")
                # Optimization measure fixed to RMSE for now
              )
          )
      ),

      div(class = "panel panel-default", style = "margin-top: 15px;",
          div(class = "panel-heading", style = "background-color: #006633; color: white;",
              h4("4. Prediction Settings", style = "margin: 0; color: white;")),
          div(class = "panel-body",
              # Add resolution slider for raster prediction
              sliderInput("raster_resolution", "Raster Resolution (meters):",
                          min = 10, max = 200, value = 30, step = 10),

              # Action button to trigger model training
              div(style = "text-align: center; margin-top: 15px;",
                  actionButton("train_button", "Train Model",
                               icon = icon("cogs"),
                               style = "background-color: #006633; color: black; font-weight: bold; width: 100%;")
              )
          )
      ),
      width = 4
    ),
    mainPanel(
      tabsetPanel(id = "main_tabs", type = "pills",
        tabPanel("Home",
                 div(class = "panel panel-default",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("MLR3 Spatial Prediction Tool", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         uiOutput("home_content")
                     )
                 )
        ),
        tabPanel("Data Preview",
                 div(class = "panel panel-default",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("Uploaded CSV Data (Head)", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         dataTableOutput("csv_table_head")
                     )
                 ),

                 div(class = "panel panel-default", style = "margin-top: 15px;",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("Uploaded Raster Files", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         verbatimTextOutput("raster_names")
                     )
                 ),

                 div(class = "panel panel-default", style = "margin-top: 15px;",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("Map of Response Variable", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         plotOutput("data_preview_map")
                     )
                 )
        ),
        tabPanel("Model Training & Results",
                 div(class = "panel panel-default",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("Model Performance Metrics", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         verbatimTextOutput("results_summary")
                     )
                 ),

                 div(class = "panel panel-default", style = "margin-top: 15px;",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("Predicted vs. Observed", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         plotOutput("results_plot")
                     )
                 )
        ),
        tabPanel("Feature Importance",
                 div(class = "panel panel-default",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("Feature Importance Plot", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         plotOutput("importance_plot")
                     )
                 )
        ),
        tabPanel("Model Validation",
                 div(class = "panel panel-default",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("Cross-Validation Results", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         verbatimTextOutput("cv_results")
                     )
                 ),

                 div(class = "panel panel-default", style = "margin-top: 15px;",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("Cross-Validation Plot", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         plotOutput("cv_plot")
                     )
                 )
        ),
        tabPanel("Raster Prediction",
                 div(class = "panel panel-default",
                     div(class = "panel-heading", style = "background-color: #006633; color: white;",
                         h4("Prediction on Raster", style = "margin: 0; color: white;")),
                     div(class = "panel-body",
                         conditionalPanel(
                            condition = "input.train_button != 0", # Only show once model is trained
                            fluidRow(
                              column(4,
                                     div(style = "margin-bottom: 15px;",
                                         checkboxInput("mask_prediction", "Apply Mask (Optional)", value = FALSE)
                                     ),
                                     conditionalPanel(
                                       condition = "input.mask_prediction == true",
                                       fileInput("mask_file", "Upload Mask File (.geojson, .shp)",
                                                 accept = c(".geojson", ".shp", ".dbf", ".prj", ".shx"))
                                     ),
                                     div(style = "margin-top: 20px; text-align: center;",
                                         actionButton("predict_button", "Generate Prediction",
                                                     icon = icon("map"),
                                                     style = "background-color: #006633; color: black; font-weight: bold; width: 100%;")
                                     ),
                                     div(style = "margin-top: 15px; text-align: center;",
                                         downloadButton("download_prediction", "Download Prediction",
                                                       style = "background-color: #004d26; color: black; font-weight: bold; width: 100%;")
                                     )
                              ),
                              column(8,
                                     plotOutput("prediction_map", height = "500px")
                              )
                            )
                         ),
                         conditionalPanel(
                            condition = "input.train_button == 0",
                            div(class = "alert alert-info", role = "alert",
                                icon("info-circle"),
                                "Please train a model first before generating predictions."
                            )
                         )
                     )
                 )
        )
      ),
      width = 8
    )
  )
)

# Define Server Logic
server <- function(input, output, session) {

  # --- Home Content ---
  # Read and render the home.md content
  output$home_content <- renderUI({
    # Read the markdown file
    home_md <- readLines("home.md")

    # Convert markdown to HTML
    html_content <- markdown::markdownToHTML(text = paste(home_md, collapse = "\n"),
                                          fragment.only = TRUE)

    # Apply custom styling to the HTML content
    styled_html <- paste0(
      "<style>",
      "h1, h2, h3 { color: #006633; }",
      "a { color: #004d26; }",
      "a:hover { color: #006633; }",
      "ul { padding-left: 20px; }",
      "</style>",
      html_content
    )

    # Return the HTML content
    HTML(styled_html)
  })

  # --- Reactive Data Loading ---

  # Reactive expression to load CSV data
  csv_data <- reactive({
    req(input$csv_file)
    tryCatch({
      fread(input$csv_file$datapath)
    }, error = function(e) {
      showNotification(paste("Error loading CSV:", e$message), type = "error")
      return(NULL)
    })
  })

  # Reactive expression to load Raster data
  raster_stack_list <- reactive({
      req(input$raster_files)
      raster_files_info <- input$raster_files
      # Need terra::rast() which works with paths directly
      rast_list <- tryCatch({
          lapply(raster_files_info$datapath, terra::rast)
      }, error = function(e) {
          showNotification(paste("Error loading raster(s):", e$message), type = "error")
          return(NULL)
      })

      # Check if all rasters loaded successfully
      if (is.null(rast_list) || any(sapply(rast_list, is.null))) {
          return(NULL)
      }

      # Optional: Check CRS consistency (basic check)
      crs_list <- sapply(rast_list, terra::crs)
      if (length(unique(crs_list)) > 1) {
          showNotification("Warning: Rasters have different CRS. Using the CRS of the first raster.", type = "warning")
          # Could attempt reprojection here if needed, but adds complexity
      }

      # Assign original names
      names(rast_list) <- raster_files_info$name

      return(rast_list)
  })

  # Combine rasters into a stack (SpatRaster)
  raster_stack <- reactive({
      req(raster_stack_list())
      rast_list <- raster_stack_list()
      tryCatch({
        # Ensure layers have unique names before stacking
        rast_names <- names(rast_list)
        if (anyDuplicated(rast_names)) {
            rast_names <- make.unique(rast_names)
            showNotification("Duplicate raster filenames detected, made names unique.", type="warning")
        }
        # Stack using c() for SpatRasters
        s <- terra::rast(rast_list)
        names(s) <- rast_names # Re-assign potentially unique names
        return(s)
      }, error = function(e) {
          showNotification(paste("Error stacking rasters:", e$message), type = "error")
          return(NULL)
      })
  })

  # --- Dynamic UI for Column Selection ---

  # Update column selectors based on uploaded CSV
  observe({
    df <- csv_data()
    if (!is.null(df)) {
      cols <- names(df)
      # Guess likely coordinate columns
      x_guess <- cols[tolower(cols) %in% c("x", "lon", "long", "longitude", "coords.x1")]
      y_guess <- cols[tolower(cols) %in% c("y", "lat", "latitude", "coords.x2")]
      # Guess response (numeric columns not coordinates)
      num_cols <- names(df)[sapply(df, is.numeric)]
      resp_guess <- setdiff(num_cols, c(x_guess, y_guess))
      resp_guess <- if(length(resp_guess) > 0) resp_guess[[1]] else NULL # Take first guess

      output$coord_x_selector <- renderUI({
        selectInput("coord_x", "Select X Coordinate Column:", choices = cols, selected = if(length(x_guess) == 1) x_guess else NULL)
      })
      output$coord_y_selector <- renderUI({
        selectInput("coord_y", "Select Y Coordinate Column:", choices = cols, selected = if(length(y_guess) == 1) y_guess else NULL)
      })
      output$response_selector <- renderUI({
        selectInput("response_var", "Select Response Variable Column:", choices = num_cols, selected = resp_guess)
      })
    } else {
      # Clear selectors if no CSV is loaded
      output$coord_x_selector <- renderUI({})
      output$coord_y_selector <- renderUI({})
      output$response_selector <- renderUI({})
    }
  })

  # --- Data Preview Outputs ---

  output$csv_table_head <- renderDataTable({
    req(csv_data())
    head(csv_data())
  })

  output$raster_names <- renderPrint({
    req(raster_stack())
    cat("Loaded Rasters:
")
    print(names(raster_stack()))
    cat("
CRS:
")
    print(crs(raster_stack())) # Show CRS
  })

  # Data preview map - shows response variable in data preview tab
  output$data_preview_map <- renderPlot({
    # Use the prepared data which includes original points and CRS
    prep_data_list <- prepared_data()
    req(prep_data_list, prep_data_list$original_data, prep_data_list$crs)

    df_orig <- prep_data_list$original_data
    x_col <- prep_data_list$x
    y_col <- prep_data_list$y
    resp_col <- prep_data_list$response
    map_crs <- prep_data_list$crs

    req(x_col %in% names(df_orig), y_col %in% names(df_orig), resp_col %in% names(df_orig))

    # Create sf object
    points_sf <- tryCatch({
        st_as_sf(df_orig, coords = c(x_col, y_col), crs = map_crs)
    }, error = function(e) {
        showNotification(paste("Error creating sf object for map:", e$message), type="error")
        return(NULL)
    })

    req(points_sf)

    # Check if response variable is numeric for continuous scale
    if (!is.numeric(points_sf[[resp_col]])) {
        showNotification("Response variable must be numeric for map coloring.", type="error")
        return(NULL)
    }

    ggplot() +
        geom_sf(data = points_sf, aes(color = .data[[resp_col]]), size = 2, alpha = 0.8) +
        scale_color_viridis_c() +
        labs(
            title = "Map of Response Variable",
            color = resp_col, # Legend title
            x = "Longitude", y = "Latitude"
        ) +
        theme_minimal()
  })


  # --- Reactive Data Preparation for mlr3 ---

  prepared_data <- reactive({
    req(csv_data(), raster_stack(), input$coord_x, input$coord_y, input$response_var)

    df <- csv_data()
    rs <- raster_stack()
    x_col <- input$coord_x
    y_col <- input$coord_y
    resp_col <- input$response_var

    # Ensure coordinate and response columns exist
    req(x_col %in% names(df), y_col %in% names(df), resp_col %in% names(df))

    # Create spatial points from CSV coordinates
    coords <- df[, c(x_col, y_col), with = FALSE] # Select columns by name

    # Check for non-numeric or NA coordinates
     if (!all(sapply(coords, is.numeric)) || anyNA(coords)) {
        showNotification("Coordinate columns must be numeric and contain no NAs.", type = "error")
        return(NULL)
    }

    pts <- tryCatch({
        # Assume coordinates are in the same CRS as the *first* raster for extraction
        # A more robust solution would involve user input for CSV CRS or checking/transforming
        vect(as.matrix(coords), crs = crs(rs)) # Create SpatVector
    }, error = function(e) {
         showNotification(paste("Error creating spatial points from coordinates:", e$message), type = "error")
         return(NULL)
    })

    if (is.null(pts)) return(NULL)


    # Extract raster values at point locations
    extracted_values <- tryCatch({
      terra::extract(rs, pts, ID = FALSE) # ID=FALSE returns only values
    }, error = function(e) {
      showNotification(paste("Error extracting raster values:", e$message), type = "error")
      return(NULL)
    })

    if (is.null(extracted_values)) return(NULL)

    # Combine extracted values with response variable and coordinates
    # Ensure column names from raster stack are valid data.table names
    valid_colnames <- make.names(names(rs))
    setnames(extracted_values, old = names(extracted_values), new = valid_colnames)

    combined_dt <- cbind(df[, .(get(resp_col), get(x_col), get(y_col))], extracted_values)
    setnames(combined_dt, old=c("V1", "V2", "V3"), new=c(resp_col, x_col, y_col)) # Rename the response/coord columns

    # Remove rows with NA values resulting from extraction (points outside rasters)
    rows_before <- nrow(combined_dt)
    combined_dt <- na.omit(combined_dt)
    rows_after <- nrow(combined_dt)
    if(rows_after < rows_before){
        showNotification(paste(rows_before - rows_after, "rows removed due to NA values after raster extraction (points likely outside raster extent)."), type = "warning")
    }

    if (nrow(combined_dt) == 0) {
        showNotification("No valid data remaining after combining and removing NAs.", type = "error")
        return(NULL)
    }


    return(list(data = combined_dt,
                x = x_col,
                y = y_col,
                response = resp_col,
                predictors = valid_colnames,
                crs = terra::crs(rs), # Include CRS for map plotting
                original_data = df # Keep original for mapping NAs if needed
                ))
  })


  # --- Model Training ---
  # Use eventReactive to trigger training only when the button is clicked
  model_results <- eventReactive(input$train_button, {
      req(prepared_data())

      prep_data_list <- prepared_data()
      data_dt <- prep_data_list$data
      resp_col <- prep_data_list$response
      coord_names <- c(prep_data_list$x, prep_data_list$y)
      predictor_names <- prep_data_list$predictors

      # Input validation before proceeding
      req(nrow(data_dt) > 0, length(predictor_names) > 0)

      # 1. Create mlr3 Task
      # Option 1: Standard Regression Task (ignores spatial nature for CV)
      task <- tryCatch({
          TaskRegr$new(id = paste0(input$model_type, "_task"),
                       backend = data_dt,
                       target = resp_col)
                       # Note: We don't explicitly set coordinate roles here for TaskRegr
      }, error = function(e) {
          showNotification(paste("Error creating mlr3 task:", e$message), type = "error")
          NULL
      })

      # Option 2: Spatial Task (if spatial CV is desired)
      # task <- tryCatch({
      #     TaskRegrST$new(id = paste0(input$model_type, "_task_st"),
      #                    backend = data_dt,
      #                    target = resp_col,
      #                    coordinate_names = coord_names,
      #                    crs = terra::crs(raster_stack())) # Pass CRS
      # }, error = function(e) {
      #     showNotification(paste("Error creating mlr3 spatial task:", e$message), type = "error")
      #     NULL
      # })

      if(is.null(task)) return(NULL)

      # 2. Define Learner (add scaling pipeline)
      learner_id <- paste0("regr.", input$model_type)
      base_learner <- lrn(learner_id)

      # Check if learner exists
      if (is.null(base_learner)) {
         showNotification(paste("Learner", learner_id, "not found. Check if package is installed (e.g., mlr3learners)."), type = "error")
         return(NULL)
      }

      graph <- po("scale") %>>% base_learner
      graph_learner <- GraphLearner$new(graph)

      # 3. Define Resampling Strategy
      # Use cross-validation with 5 folds for validation tab
      cv_resampling <- rsmp("cv", folds = 5)

      # Keep holdout for main model training
      train_prop <- input$train_split_ratio
      resampling <- rsmp("holdout", ratio = train_prop)

      # Option 2: Spatial CV (requires TaskRegrST)
      # resampling <- rsmp("spcv_coords", folds = 5)

      # --- Hyperparameter Tuning (Conditional) ---
      if (input$enable_tuning) {
        showNotification("Starting hyperparameter tuning...", id="tune_msg", duration = NULL)

        search_space <- NULL
        # Define search space based on model type
        n_features = length(prep_data_list$predictors)
        if (input$model_type == "ranger") {
          search_space = ps(
            regr.ranger.mtry = p_int(lower = 1, upper = max(1, n_features)),
            regr.ranger.min.node.size = p_int(lower = 1, upper = 10)
            # regr.ranger.num.trees = p_int(lower = 50, upper = 500)
          )
        } else if (input$model_type == "gbm") {
          search_space = ps(
              regr.gbm.n.trees = p_int(lower = 50, upper = 500),
              regr.gbm.interaction.depth = p_int(lower = 1, upper = 5),
              regr.gbm.shrinkage = p_dbl(lower = 0.01, upper = 0.3)
          )
        }

        if (is.null(search_space)) {
          showNotification("Tuning not configured for this model type.", type="warning")
        } else {
          terminator = trm("evals", n_evals = input$tuning_evals)
          tuner = tnr("random_search") # Fixed tuner
          inner_resampling = rsmp("cv", folds = 3) # Fixed inner resampling
          measure = msr("regr.rmse") # Fixed measure for optimization

          # Instantiate the main holdout split to get training indices
          resampling$instantiate(task)
          train_idx <- resampling$train_set(1)

          instance = TuningInstanceBatchSingleCrit$new(
            task = task,
            learner = graph_learner,
            resampling = inner_resampling, # Use inner CV
            measure = measure,
            search_space = search_space,
            terminator = terminator,
            store_models = FALSE # Don't store inner models
          )

          # Run tuning *only on the training set*
          tuner$optimize(instance)

          # Check if tuning results exist
          if (!is.null(instance$result_learner_param_vals)) {
               # Set the best found hyperparameters on the learner for the final evaluation
               showNotification(paste("Tuning complete. Best params found:", paste(names(instance$result_learner_param_vals), instance$result_learner_param_vals, sep = " = ", collapse = ", ")), type="message")
               graph_learner$param_set$values <- modifyList(
                   graph_learner$param_set$values, # Keep existing fixed params (like from graph)
                   instance$result_learner_param_vals
               )
          } else {
              showNotification("Tuning finished, but no results found. Using default hyperparameters.", type="warning")
          }
        }
         removeNotification(id="tune_msg")
      } # End if(input$enable_tuning)

      # 4. Run Resampling (Holdout evaluation with potentially tuned learner)
      # Use a progress indicator
      eval_msg <- ifelse(input$enable_tuning, "Evaluating tuned model on test set...", "Evaluating model on test set...")
      showNotification(eval_msg, type = "message", duration = NULL, id="eval_msg")
      rr <- tryCatch({
          resample(task, graph_learner, resampling, store_models = FALSE) # store_models = FALSE for efficiency
      }, error = function(e) {
          removeNotification(id="eval_msg")
          showNotification(paste("Error during model resampling:", e$message), type = "error")
          return(NULL)
      })
      removeNotification(id="eval_msg") # Remove progress message

      if(is.null(rr)) return(NULL)
      showNotification("Model evaluation complete! Training final model on full dataset for importance...", type = "message", id="final_model_msg")

      # 5. Aggregate Performance
      measures <- msrs(c("regr.rmse", "regr.rsq", "regr.mae"))
      aggregated_perf <- rr$aggregate(measures)

      # 6. Train final model on full task for importance calculation
      importance_scores <- NULL
      final_model_error <- NULL
      learner_id <- paste0("regr.", input$model_type) # Define learner ID used in graph

      # Configure model-specific params for importance before training
      if (input$model_type == "ranger") {
          # Set importance calculation for ranger
          graph_learner$param_set$values$regr.ranger.importance <- "impurity"
      } else if (input$model_type == "gbm") {
          # Ensure GBM importance can be extracted
          graph_learner$param_set$values$regr.gbm.keep.data <- TRUE
      }

      tryCatch({
          # Train on the full task to enable importance extraction
          graph_learner$train(task)

          # Model-specific extraction of importance values
          if (input$model_type == "ranger") {
              # Get the ranger model from the pipeline
              trained_graph_step <- graph_learner$model[[learner_id]]
              if(!is.null(trained_graph_step) && !is.null(trained_graph_step$model) &&
                 "variable.importance" %in% names(trained_graph_step$model)) {
                  importance_scores <- trained_graph_step$model$variable.importance
              } else {
                  # Ensure importance was activated in ranger
                  final_model_error <- "Could not extract variable importance from Random Forest."
              }
          } else if (input$model_type == "gbm") {
              trained_graph_step <- graph_learner$model[[learner_id]]
              if(!is.null(trained_graph_step) && !is.null(trained_graph_step$model)) {
                  # gbm::summary returns a data frame with variables and relative influence
                  imp_summary <- tryCatch(
                      summary(trained_graph_step$model, plotit = FALSE),
                      error = function(e) NULL
                  )
                  if (!is.null(imp_summary) && nrow(imp_summary) > 0) {
                      importance_scores <- setNames(imp_summary$rel.inf, as.character(imp_summary$var))
                  } else {
                      final_model_error <- "Could not extract variable importance from GBM model."
                  }
              }
          } else {
              # Try to use mlr3's built-in importance extractor as fallback
              if ("importance" %in% graph_learner$properties) {
                  importance_scores <- graph_learner$importance()
              } else {
                  final_model_error <- paste("Importance extraction not configured for", input$model_type)
              }
          }

          # If we still don't have importance scores, report the error
          if (is.null(importance_scores) && is.null(final_model_error)) {
              final_model_error <- paste("Unable to extract importance for", input$model_type)
          }
      }, error = function(e) {
          final_model_error <- paste("Error training final model or getting importance:", e$message)
      })

      removeNotification(id="final_model_msg")

      if (!is.null(final_model_error)) {
          showNotification(final_model_error, type="error")
      } else if (is.null(importance_scores)) {
           showNotification("Feature importance not available for the selected model.", type = "warning")
      } else {
          showNotification("Final model trained and importance calculated.", type = "message")
      }

      # 7. Run cross-validation for model validation tab
      showNotification("Running 5-fold cross-validation...", type = "message", id="cv_msg")
      cv_results <- tryCatch({
          # We use the same learner but with cross-validation resampling
          cv_rr <- resample(task, graph_learner, cv_resampling, store_models = FALSE)

          # Calculate performance measures for CV
          cv_perf <- cv_rr$aggregate(measures)

          # Get all CV predictions for plotting
          cv_preds <- cv_rr$prediction()

          # Return CV results
          list(
              performance = cv_perf,
              predictions = cv_preds,
              fold_performances = cv_rr$score(measures)
          )
      }, error = function(e) {
          showNotification(paste("Error during cross-validation:", e$message), type = "error")
          return(NULL)
      })
      removeNotification(id="cv_msg")

      # Return results needed for output
      return(list(
          performance = aggregated_perf,
          predictions = rr$prediction(), # Get predictions for plotting
          importance = importance_scores, # Add importance scores
          cv_results = cv_results # Add cross-validation results
      ))
  })


  # --- Results Outputs ---

  output$results_summary <- renderPrint({
    results <- model_results()
    req(results)
    print(results$performance)
  })

  output$results_plot <- renderPlot({
    results <- model_results()
    req(results)

    preds <- as.data.table(results$predictions)
    req(nrow(preds) > 0) # Ensure predictions exist

    ggplot(preds, aes(x = truth, y = response)) +
      geom_point(alpha = 0.6) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
      labs(
        title = paste("Predicted vs. Observed -", toupper(input$model_type)),
        x = "Observed Response",
        y = "Predicted Response"
      ) +
      theme_minimal() +
      coord_equal() # Ensure axes have the same scale for 1:1 line interpretation
  })

  # --- Feature Importance Plot ---
  output$importance_plot <- renderPlot({
      results <- model_results()
      prep_data_list <- prepared_data()
      req(results, prep_data_list)

      # Get coordinate column names to exclude them from importance plot
      coord_cols <- c(prep_data_list$x, prep_data_list$y)

      if (!is.null(results$importance) && length(results$importance) > 0) {
          # Convert importance values to data frame for plotting
          imp_data <- data.frame(
              Variable = names(results$importance),
              Importance = as.numeric(results$importance)
          )

          # Make sure we have valid importance data
          if (nrow(imp_data) == 0 || all(is.na(imp_data$Importance))) {
              showNotification("Unable to generate variable importance plot: No valid importance values", type = "error")
              return(NULL)
          }

          # Remove coordinate columns from importance data
          imp_data <- imp_data[!imp_data$Variable %in% coord_cols, ]

          # Remove features with zero or negative importance if any
          imp_data <- imp_data[imp_data$Importance > 0, ]

          if (nrow(imp_data) == 0) {
              showNotification("No positive importance values found to plot", type = "warning")
              return(NULL)
          }

          # Sort by importance in descending order
          importance_df <- imp_data[order(imp_data$Importance, decreasing = TRUE), ]

          # Normalize importance values if they're very large
          if (max(importance_df$Importance) > 1000) {
              importance_df$Importance <- importance_df$Importance / sum(importance_df$Importance) * 100
              y_label <- "Relative Importance (%)"
          } else if (input$model_type == "ranger") {
              y_label <- "Importance (Impurity Reduction)"
          } else if (input$model_type == "gbm") {
              y_label <- "Relative Influence (%)"
          } else {
              y_label <- "Importance"
          }

          # Limit to top 20 variables if there are many
          if (nrow(importance_df) > 20) {
              importance_df <- importance_df[1:20, ]
              showNotification("Plotting top 20 most important variables only", type = "message")
          }

          # Create the plot
          ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
              geom_bar(stat = "identity", fill = "steelblue") +
              coord_flip() +
              theme_minimal() +
              labs(
                  title = paste("Variable Importance -", toupper(input$model_type)),
                  x = "Predictor Variables",
                  y = y_label
              ) +
              theme(
                  plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
                  axis.text = element_text(size = 10),
                  axis.title = element_text(size = 12)
              )
      } else {
          # Display placeholder message when no importance values are available
          plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "")
          text(0, 0, "Variable importance not available for this model configuration", cex = 1.2)
          return(NULL)
      }
  })

  # --- Cross-Validation Results ---
  output$cv_results <- renderPrint({
    results <- model_results()
    req(results, results$cv_results)

    # Print CV performance metrics
    cat("Cross-Validation Performance Metrics (5-fold):\n")
    print(results$cv_results$performance)

    # Print individual fold performances
    cat("\nPerformance by Fold:\n")
    fold_data <- results$cv_results$fold_performances
    # Select only numeric columns for better display
    numeric_cols <- sapply(fold_data, is.numeric)
    print(fold_data[, numeric_cols, with = FALSE])
  })

  output$cv_plot <- renderPlot({
    results <- model_results()
    req(results, results$cv_results)

    cv_preds <- as.data.table(results$cv_results$predictions)
    req(nrow(cv_preds) > 0) # Ensure predictions exist

    # Add fold information for coloring
    cv_preds$fold <- rep(1:5, each = ceiling(nrow(cv_preds)/5))[1:nrow(cv_preds)]

    # Create scatterplot of predicted vs observed values, colored by fold
    ggplot(cv_preds, aes(x = truth, y = response, color = factor(fold))) +
      geom_point(alpha = 0.6) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
      labs(
        title = paste("Cross-Validation Results (5-fold) -", toupper(input$model_type)),
        x = "Observed Response",
        y = "Predicted Response",
        color = "Fold"
      ) +
      scale_color_viridis_d() +
      theme_minimal() +
      coord_equal() # Ensure axes have the same scale for 1:1 line interpretation
  })

  # --- Predicted Data Map ---
  output$predicted_map_plot <- renderPlot({
      req(prediction_raster())  # Ensure prediction raster is available

      pred_rast <- prediction_raster()

      ggplot() +
          tidyterra::geom_spatraster(data = pred_rast) +
          scale_fill_viridis_c(name = "Predicted Values", na.value = "transparent") +
          labs(
              title = "Map of Predicted Response Variable",
              x = "Longitude", y = "Latitude"
          ) +
          theme_minimal()
  })

  # --- Raster Prediction ---

  # Store prediction raster in a reactive value for downloading
  prediction_raster <- reactiveVal(NULL)

  # Generate prediction when button is clicked
  observeEvent(input$predict_button, {
    # Make sure we have a trained model
    results <- model_results()
    req(results, raster_stack())

    # Show a progress notification
    showNotification("Generating prediction map...", id = "pred_progress", duration = NULL)

    # Get the trained model and task information
    tryCatch({
      # Access the full model stored in the results object
      learner_id <- paste0("regr.", input$model_type)
      graph_learner <- NULL

      # First we need to prepare the trained model - we'll use the one from the importance calculation
      # that was already trained on the full task
      prep_data_list <- prepared_data()
      task <- TaskRegr$new(
        id = paste0(input$model_type, "_predict_task"),
        backend = prep_data_list$data,
        target = prep_data_list$response
      )

      # Create graph learner as done in model_results
      base_learner <- lrn(learner_id)
      graph <- po("scale") %>>% base_learner
      graph_learner <- GraphLearner$new(graph)

      # Configure model-specific params (same as for importance)
      if (input$model_type == "ranger") {
        graph_learner$param_set$values$regr.ranger.importance <- "impurity"
      } else if (input$model_type == "gbm") {
        graph_learner$param_set$values$regr.gbm.keep.data <- TRUE
      }

      # Train on full dataset
      graph_learner$train(task)

      # Get the feature names used for prediction (excluding coordinates and target)
      feature_names <- setdiff(task$feature_names, c(prep_data_list$x, prep_data_list$y, prep_data_list$response))

      # Get the raster stack
      raster_stack_input <- raster_stack()

      # Adjust raster resolution
      resolution <- input$raster_resolution
      raster_stack_input <- tryCatch({
        terra::aggregate(raster_stack_input, fact = resolution / terra::res(raster_stack_input)[1])
      }, error = function(e) {
        showNotification(paste("Error adjusting raster resolution:", e$message), type = "error")
        return(NULL)
      })

      # Prepare mask if provided
      mask_poly <- NULL
      if (input$mask_prediction && !is.null(input$mask_file)) {
        # Read mask file (GeoPackage or GeoJSON)
        mask_poly <- tryCatch({
          vect(input$mask_file$datapath)
        }, error = function(e) {
          showNotification(paste("Error reading mask file:", e$message), type = "error")
          return(NULL)
        })
      }

      # Process raster in chunks to reduce memory usage
      showNotification("Preparing raster data for prediction...", type = "message", id = "prep_msg")

      # Check raster size to determine processing approach
      pixel_count <- terra::ncell(raster_stack_input)

      # For smaller rasters (less than ~1 million cells), process all at once
      if (pixel_count <= 1000000) {
        raster_df <- tryCatch({
          as.data.frame(raster_stack_input, xy = TRUE, na.rm = FALSE)
        }, error = function(e) {
          showNotification(paste("Error converting raster to data frame:", e$message), type = "error")
          removeNotification(id = "pred_progress")
          return(NULL)
        })

        if (is.null(raster_df) || nrow(raster_df) == 0) {
          showNotification("No valid data extracted from raster stack.", type = "error")
          removeNotification(id = "pred_progress")
          return(NULL)
        }

        # Set coordinate column names
        names(raster_df)[1:2] <- c("x", "y")

        # Convert to data.table for consistency
        raster_dt <- as.data.table(raster_df)

        # Create temporary target column (required for task creation but not used)
        first_col <- names(raster_dt)[3]  # First column after x,y
        raster_dt[, temp_target := get(first_col)]

        # Create prediction task
        pred_task <- TaskRegr$new(
          id = "raster_prediction_task",
          backend = raster_dt,
          target = "temp_target"
        )

        # Make predictions
        showNotification("Running prediction...", type = "message", id = "predict_msg")
        predictions <- graph_learner$predict(pred_task)
        raster_dt$predicted_value <- predictions$response

      } else {
        # For larger rasters, process in chunks to conserve memory
        showNotification(paste("Large raster detected (", format(pixel_count, big.mark=","),
                              " pixels). Processing in chunks..."), type = "message")

        # Create an empty template raster for results with same dimensions as first layer
        pred_template <- rast(raster_stack_input[[1]])
        terra::values(pred_template) <- NA

        # Determine chunk size (adjust based on system memory)
        chunk_size <- 100000  # Number of cells per chunk
        n_chunks <- ceiling(pixel_count / chunk_size)

        # Create progress bar
        showNotification(paste("Processing raster in", n_chunks, "chunks..."),
                       type = "message", id = "chunk_progress")

        # Process each chunk
        for (i in 1:n_chunks) {
          # Update progress
          showNotification(paste("Processing chunk", i, "of", n_chunks),
                         type = "message", id = "current_chunk")

          # Calculate chunk indices
          start_idx <- (i-1) * chunk_size + 1
          end_idx <- min(i * chunk_size, pixel_count)

          # Extract chunk cells
          chunk_cells <- terra::cells(raster_stack_input)
          chunk_cells <- chunk_cells[start_idx:end_idx]

          # Extract values and coordinates for this chunk
          chunk_df <- tryCatch({
            values <- terra::extract(raster_stack_input, chunk_cells)
            coords <- terra::xyFromCell(raster_stack_input, chunk_cells)
            cbind(coords, values)
          }, error = function(e) {
            showNotification(paste("Error extracting chunk", i, ":", e$message), type = "warning")
            return(NULL)
          })

          if (!is.null(chunk_df) && nrow(chunk_df) > 0) {
            # Set column names
            names(chunk_df)[1:2] <- c("x", "y")

            # Convert to data.table
            chunk_dt <- as.data.table(chunk_df)

            # Create temporary target
            first_col <- names(chunk_dt)[3]
            chunk_dt[, temp_target := get(first_col)]

            # Create prediction task
            chunk_task <- TaskRegr$new(
              id = paste0("chunk_task_", i),
              backend = chunk_dt,
              target = "temp_target"
            )

            # Make predictions for this chunk
            chunk_pred <- tryCatch({
              preds <- graph_learner$predict(chunk_task)
              chunk_dt$predicted_value <- preds$response
              chunk_dt
            }, error = function(e) {
              showNotification(paste("Error predicting chunk", i, ":", e$message), type = "warning")
              return(NULL)
            })

            # Update the prediction template raster with chunk results
            if (!is.null(chunk_pred) && "predicted_value" %in% names(chunk_pred)) {
              # Update values in the template raster
              terra::setValues(pred_template, values = chunk_pred$predicted_value,
                             cells = chunk_cells)
            }
          }
        }

        # Create data.table with all predicted values for returning
        raster_dt <- tryCatch({
          # Convert the final predicted raster to data.frame with coordinates
          as.data.table(as.data.frame(pred_template, xy = TRUE, na.rm = TRUE))
        }, error = function(e) {
          showNotification(paste("Error creating final prediction data:", e$message), type = "error")
          removeNotification(id = "pred_progress")
          removeNotification(id = "chunk_progress")
          removeNotification(id = "current_chunk")
          return(NULL)
        })

        # Change column names to match expected format
        if (!is.null(raster_dt) && ncol(raster_dt) >= 3) {
          names(raster_dt)[3] <- "predicted_value"
        } else {
          showNotification("Failed to create valid prediction data", type = "error")
          removeNotification(id = "pred_progress")
          removeNotification(id = "chunk_progress")
          removeNotification(id = "current_chunk")
          return(NULL)
        }

        removeNotification(id = "chunk_progress")
        removeNotification(id = "current_chunk")
      }

      # Show completion message
      showNotification("Prediction calculation complete, creating raster...", type = "message", id = "predict_msg")

      # Convert predictions back to a raster
      pred_rast <- tryCatch({
        # Create the prediction raster using terra
        pred_xyz <- raster_dt[, .(x, y, predicted_value)]
        # Remove rows with NA predictions
        pred_xyz <- na.omit(pred_xyz)

        # Create the raster, matching the CRS and resolution of the original
        prediction_r <- rast(
          x = pred_xyz,
          crs = crs(raster_stack_input),
          type = "xyz"
        )

        # Mask prediction if requested and mask file is valid
        if (!is.null(mask_poly)) {
          prediction_r <- mask(prediction_r, mask_poly)
        }

        prediction_r
      }, error = function(e) {
        showNotification(paste("Error creating prediction raster:", e$message), type = "error")
        return(NULL)
      })

      # Store the result
      prediction_raster(pred_rast)

      # Final notification
      if (!is.null(pred_rast)) {
        showNotification("Prediction completed successfully!", type = "message")
      } else {
        showNotification("Failed to create prediction raster", type = "error")
      }

      # Clean up progress notifications
      removeNotification(id = "pred_progress")
      removeNotification(id = "prep_msg")
      removeNotification(id = "predict_msg")

    }, error = function(e) {
      showNotification(paste("Error in prediction process:", e$message), type = "error")
      removeNotification(id = "pred_progress")
    })
  })

  # Render the prediction map
  output$prediction_map <- renderPlot({
    req(prediction_raster())

    # Create a visualization of the prediction raster
    pred_rast <- prediction_raster()

    # Check if mask is available
    mask_poly <- NULL
    if (input$mask_prediction && !is.null(input$mask_file)) {
      mask_poly <- tryCatch({
        st_read(input$mask_file$datapath)
      }, error = function(e) {
        return(NULL)
      })
    }

    # Make sure we have the original raster for extent reference
    orig_rast <- raster_stack()
    mask_poly_sf <- NULL
    if (!is.null(mask_poly)) {
      mask_poly_sf <- st_as_sf(mask_poly)
    }

    # Get the proper extent from original rasters
    if (!is.null(orig_rast)) {
      # Get the extent from the first layer of the stack
      bbox <- ext(orig_rast)

      # Create the plot with correct extent
      p <- ggplot() +
        tidyterra::geom_spatraster(data = pred_rast) +
        scale_fill_viridis_c(name = paste("Predicted", input$response_var),
                            option = "viridis",
                            na.value = "transparent") +
        theme_minimal() +
        labs(title = paste("Spatial Prediction of", input$response_var),
             x = "X Coordinate", y = "Y Coordinate") +
        theme(legend.position = "right") +
        # Force the plot to use the original raster extent
        coord_sf(xlim = c(bbox[1], bbox[2]), ylim = c(bbox[3], bbox[4]))

      # Add the mask boundary if available
      if (!is.null(mask_poly_sf)) {
        p <- p + geom_sf(data = mask_poly_sf, fill = NA, color = "black", linewidth = 0.5)
      }
    } else {
      # Fallback if original raster isn't available
      p <- ggplot() +
        tidyterra::geom_spatraster(data = pred_rast) +
        scale_fill_viridis_c(name = paste("Predicted", input$response_var),
                            option = "viridis",
                            na.value = "transparent") +
        theme_minimal() +
        labs(title = paste("Spatial Prediction of", input$response_var),
             x = "X Coordinate", y = "Y Coordinate") +
        theme(legend.position = "right")

      # Add the mask boundary if available
      if (!is.null(mask_poly_sf)) {
        p <- p + geom_sf(data = mask_poly_sf, fill = NA, color = "black", linewidth = 0.5)
      }
    }

    p
  })

  # Download handler for the prediction raster
  output$download_prediction <- downloadHandler(
    filename = function() {
      paste0("prediction_", input$model_type, "_", format(Sys.time(), "%Y%m%d_%H%M"), ".tif")
    },
    content = function(file) {
      req(prediction_raster())

      # Save the raster to the file
      tryCatch({
        writeRaster(prediction_raster(), file, overwrite = TRUE)
        showNotification("Prediction raster saved successfully!", type = "message")
      }, error = function(e) {
        showNotification(paste("Error saving raster:", e$message), type = "error")
      })
    },
    contentType = "image/tiff"
  )

 }

# Run the application
shinyApp(ui = ui, server = server)
