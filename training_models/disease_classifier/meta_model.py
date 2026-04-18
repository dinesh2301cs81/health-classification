model_meta = Sequential([
    Dense(16, activation='relu', input_shape=(6,)),
    Dense(8, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model_meta.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_meta.fit(meta_X, y_train, epochs=50)