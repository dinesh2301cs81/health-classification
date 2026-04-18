base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))

for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

output = Dense(3, activation='softmax')(x)

image_model = Model(inputs=base_model.input, outputs=output)

image_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

image_model.fit(X_img_train, y_train, epochs=10, batch_size=32, validation_split=0.2)