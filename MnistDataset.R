library(keras)
mnist = dataset_mnist()
train_images = mnist$train$x
train_labels = mnist$train$y
test_images = mnist$test$x
test_labels = mnist$test$y

#digit = train_images[4,,]
#plot(as.raster(digit, max = 255))

train_images = array_reshape(train_images, c(60000, 28*28))
train_images = train_images / 255

test_images = array_reshape(test_images, c(10000, 28*28))
test_images = test_images/255

train_labels = to_categorical(train_labels, num_classes = 11)
test_labels = to_categorical(test_labels, num_classes = 11)


network = keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", initializer_he_normal(),input_shape = c(28*28)) %>%
  layer_dense(units = 16, activation = "relu", initializer_he_normal()) %>% 
  layer_dense(units = 11, activation = "softmax")

network %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

network

network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128, validation_split = .2)

metrics = network %>% evaluate(test_images, test_labels)
metrics

network %>% predict_classes(test_images[1:10,])