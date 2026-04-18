model_spo2 = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(2, activation='softmax')
])

y_spo2 = (spo2_train.flatten() < 94).astype(int)

model_spo2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_spo2.fit(spo2_train, y_spo2, epochs=30)