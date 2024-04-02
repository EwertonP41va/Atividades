import tensorflow as tf

# Carregar os dados da base de dados CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalizar os dados
x_train, x_test = x_train / 255.0, x_test / 255.0

# Definir o modelo de rede neural profunda
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=5)

# Avaliar o modelo
model.evaluate(x_test, y_test)