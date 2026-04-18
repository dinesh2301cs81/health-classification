model_hr = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(2, activation='softmax')
])

y_hr = ((hr_train.flatten() > 100) | (hr_train.flatten() < 60)).astype(int)

model_hr.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_hr.fit(hr_train, y_hr, epochs=30)