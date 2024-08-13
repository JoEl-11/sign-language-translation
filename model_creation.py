import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# path for the dataset
data_path = r"C:\Users\HP\Desktop\final project\dataforprocessing\dataimage"


classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','skip','space','T','U','V','W','X','Y','Z']  


# Paths for train, valid, and test sets
train_path = os.path.join(data_path, 'train')
valid_path = os.path.join(data_path, 'valid')
test_path = os.path.join(data_path, 'test')

# Creating directories for train, valid, and test sets
for path in [train_path, valid_path, test_path]:
    if not os.path.exists(path):
        os.makedirs(path)
        for class_name in classes:
            os.makedirs(os.path.join(path, class_name), exist_ok=True)


train_split = 0.7
valid_split = 0.2
test_split = 0.1


def split_data(class_name):
    class_dir = os.path.join(data_path, class_name)
    images = os.listdir(class_dir)
    random.shuffle(images)

    train_size = int(train_split * len(images))
    valid_size = int(valid_split * len(images))

    train_images = images[:train_size]
    valid_images = images[train_size:train_size + valid_size]
    test_images = images[train_size + valid_size:]

    return train_images, valid_images, test_images

# Moving files into the test,valid,train folders
for class_name in classes:
    train_images, valid_images, test_images = split_data(class_name)

    for image in train_images:
        shutil.move(os.path.join(data_path, class_name, image), os.path.join(train_path, class_name, image))

    for image in valid_images:
        shutil.move(os.path.join(data_path, class_name, image), os.path.join(valid_path, class_name, image))

    for image in test_images:
        shutil.move(os.path.join(data_path, class_name, image), os.path.join(test_path, class_name, image))



#normalization process
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# Creating Batches
train_batches = train_datagen.flow_from_directory(
    directory=train_path, target_size=(128, 128), classes=classes, batch_size=32, class_mode='categorical',color_mode='grayscale'
)
valid_batches = valid_datagen.flow_from_directory(
    directory=valid_path, target_size=(128, 64), classes=classes, batch_size=32, color_mode='grayscale'
)
test_batches = test_datagen.flow_from_directory(
    directory=test_path, target_size=(128, 128), classes=classes, batch_size=32, color_mode='grayscale',shuffle=False
)
 
 #creating the cnn model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(28, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
model.fit(train_batches, epochs=30, validation_data=valid_batches,callbacks=[early_stopping])

model.save('rizwinmodel3.keras')