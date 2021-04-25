from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.xception import Xception


# Load df:
    #train_path = 'PATH'
    #target_df = pd.read_csv('PATH')

for i in range(len(target_df['id'])):
    target_df['id'][i] = target_df['id'][i] + '.jpg'


# Parameters:
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
IMAGE_CHANNELS = 3
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)


# Split train & validation:
train_df, val_df = train_test_split(target_df,
                                    test_size=0.15,
                                    random_state=42)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


# ImageDataGenerator & preprocessing:
train_datagen = ImageDataGenerator(rotation_range=40,
                                   rescale=1./255.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2)

val_datagen = ImageDataGenerator(rescale=1./255.)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    train_path,
    x_col='id',
    y_col='breed',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode='sparse',
    batch_size=32)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    train_path,
    x_col='id',
    y_col='breed',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    class_mode='sparse',
    batch_size=32)


# Prepare base model:
base_model = Xception(input_shape=IMAGE_SIZE,
                      include_top=False,
                      weights="imagenet")

for layer in base_model.layers:
    layer.trainable = False

avg = keras.layers.GlobalAveragePooling2D()(base_model.output)

avg = keras.layers.Dense(512, activation='relu')(avg)

avg = keras.layers.Dropout(0.2)(avg)

output = keras.layers.Dense(120, activation="softmax")(avg)

model = keras.Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='acc')


# Callbacks
cb_model = keras.callbacks.ModelCheckpoint("8_cb_model.h5")

best_acc_model = keras.callbacks.ModelCheckpoint(
    "8_best_acc_model.h5", monitor='val_acc', mode='auto', verbose=1, save_best_only=True)

best_loss_model = keras.callbacks.ModelCheckpoint(
    "8_best_loss_model.h5", monitor='val_loss', mode='auto', verbose=1, save_best_only=True)

earlystop_model = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=5,
                                verbose=1,
                                restore_best_weights=True)


# Training:
history = model.fit(train_gen,
                   validation_data=val_gen,
                   epochs=1,
                   callbacks=[cb_model, best_acc_model])


# Prediction:
smallimage = cv2.resize((cv2.imread(train_path + target_df['id'][3])), (200, 200))

plt.imshow(cv2.imread(train_path + target_df['id'][3]))

smallimage = smallimage / 255.

smallimage = smallimage.reshape(1, 200, 200, 3)

best = model.predict(smallimage)

best.argmax()
