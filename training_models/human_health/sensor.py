sensor_model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

sensor_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

sensor_model.fit(X_sensor_train, y_train, epochs=20, batch_size=16)