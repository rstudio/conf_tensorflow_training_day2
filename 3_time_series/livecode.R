library(M4comp2018)
library(tidyverse)
library(tsibble)
library(ggplot2)
library(keras)

# See ?M4comp2018 for description of dataset
data(M4)

# Get indices for all hourly series
indices_hourly <- M4 %>%
  map_lgl(~ levels(.x$period)[[.x$period]] == "Hourly") %>%
  which()

# Pick a random one
set.seed(42)
awesome_index <- sample(indices_hourly, 1)
best_time_series_ever <- M4[[awesome_index]]

# Train/test split
training <- best_time_series_ever$x %>%
  as_tsibble() %>%
  mutate(.id = "train")
test <- best_time_series_ever$xx %>%
  as_tsibble() %>%
  mutate(.id = "test")
all_data <- rbind(training, test)

# Plot to check for reasonability
ggplot(training) + geom_line(aes(index, value))

training_with_lags <- training %>%
  mutate(lags = map(
    seq_along(value),
    function(i) {
      start <- max(0, i - 48)
      value[start:(i - 1)]
    }
  ))

# Define model
model_1 <- keras_model_sequential() %>%
  layer_lstm(128, input_shape = c(48, 1)) %>%
  layer_dense(1)

model_1 %>%
  compile(loss = "mse", optimizer = "adam")

y <- training_with_lags %>%
  slice(49:nrow(training_with_lags)) %>%
  pull(value)
x <- training_with_lags %>%
  slice(49:nrow(training_with_lags)) %>%
  pull(lags) %>%
  array_reshape(c(nrow(training_with_lags) - 48, 48, 1))

# Fit model
history <- model_1 %>%
  fit(
    x = x,
    y = y,
    batch_size = 64,
    epochs = 200
  )

# Predictor data for first value in test set
test_x <- best_time_series_ever$x %>% 
  tail(48) %>%
  array_reshape(c(1, 48, 1))

# Forecast future values given model
forecast_future <- function(model, x, steps_ahead = 48) {
  if (steps_ahead > 0) {
    prediction <- model %>%
      predict(x)
    
    x <- lead(as.vector(x), default = prediction) %>%
      array_reshape(c(1, 48, 1))
    
    steps_ahead <- steps_ahead - 1
    Recall(model, x, steps_ahead)
  } else {
    x
  }
}

predictions <- forecast_future(model_1, test_x) %>%
  as.vector()

test_with_preds <- test %>%
  mutate(prediction = predictions)

ggplot(training) + geom_line(aes(index, value)) +
  geom_line(aes(index, prediction, color = "prediction"), 
            data = test_with_preds) +
  geom_line(aes(index, value, color = "actual"), 
            data = test_with_preds) +
  NULL

# We also briefly talked about, but didn't have time to implement, other types of architectures, such as sequence-to-sequence 
#  encoder-decoder models. In those cases your model output would have shapes that accommodate the timesteps and the data prep
#  would also need to be adjusted acccordingly.

