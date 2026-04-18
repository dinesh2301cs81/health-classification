model_fever = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),
    Dense(4, activation='relu'),
    Dense(2, activation='softmax')
])

y_fever = (temp_train.flatten() > 37.5).astype(int)

model_fever.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_fever.fit(np.hstack([temp_train, hr_train]), y_fever, epochs=30)