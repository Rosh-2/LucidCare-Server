import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# --- 1. Hyperparameters ---
batch_size = 16
num_epochs = 10
img_size = (224, 224)
num_classes = 2
learning_rate = 1e-4

# --- 2. Data generators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\rejir\\OneDrive\\Desktop\\DataSet\\chest_xray\\train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    "C:\\Users\\rejir\\OneDrive\\Desktop\\DataSet\\chest_xray\\val",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# --- 3. Load EfficientNetB0 ---
base_model = densenet121(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze base model layers to speed up CPU training
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# --- 4. Compile model ---
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 5. Train model ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=num_epochs
)

# --- 6. Save model ---
model.save('efficientnetb0_xray_keras.h5')

# --- 7. Evaluate ---
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
