
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


print("TF version: ", tf.__version__)


IS_FASHION_MNIST = False


#  Model code
IMG_SIZE = 28


class Model(tf.Module):
    def __init__(self):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(
                    input_shape=(IMG_SIZE, IMG_SIZE), name="flatten"
                ),
                tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
                tf.keras.layers.Dense(10, name="dense_2"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        )

    # The `train` function takes a batch of input images and labels.
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
            tf.TensorSpec([None, 10], tf.float32),
        ]
    )
    def train(self, x, y):
        with tf.GradientTape() as tape:
            prediction = self.model(x)
            loss = self.model.loss(y, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        result = {"loss": loss}
        return result

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
        ]
    )
    def infer(self, x):
        logits = self.model(x)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {"output": probabilities, "logits": logits}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path,
            tensor_names=tensor_names,
            data=tensors_to_save,
            name="save",
        )
        return {"checkpoint_path": checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path,
                tensor_name=var.name,
                dt=var.dtype,
                name="restore",
            )
            var.assign(restored)
            restored_tensors[var.name] = restored

        return restored_tensors


# Data Loader
dataset = ""
if IS_FASHION_MNIST:
    dataset = tf.keras.datasets.fashion_mnist
else:
    dataset = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

# Scailing
train_images = (train_images / 255.0).astype(np.float32)
test_images = (test_images / 255.0).astype(np.float32)

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)


# Training code
NUM_EPOCHS = 15
BATCH_SIZE = 100

epochs = np.arange(1, NUM_EPOCHS + 1, 1)
loss = np.zeros([NUM_EPOCHS])
m = Model()

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.batch(BATCH_SIZE)


from tqdm import tqdm

for i in tqdm(range(NUM_EPOCHS)):

    for x, y in train_dataset:
        result = m.train(x, y)

    loss[i] = result["loss"]

    if (i + 1) % 10 == 0:
        print(f"Finished {i+1} epochs")
        print(f"  loss: {loss[i]:.3f}")

    # Save the trained weights to a checkpoint.
    m.save("/tmp/model.ckpt")


# Plot result
plt.plot(epochs, loss, label="Pre-training")
plt.ylim([0, max(plt.ylim())])
plt.xlabel("Epoch")
plt.ylabel("Loss [Cross Entropy]")
plt.legend()

plt.show()


# Convert the model to tflite
# 모델은 pre-trained된 (weight값이 어느정도 정해진) 모델을 넣어야 한다.

SAVED_MODEL_DIR = "saved_model"

tf.saved_model.save(
    m,
    SAVED_MODEL_DIR,
    signatures={
        "train": m.train.get_concrete_function(),
        "infer": m.infer.get_concrete_function(),
        "save": m.save.get_concrete_function(),
        "restore": m.restore.get_concrete_function(),
    },
)


converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]
converter.experimental_enable_resource_variables = True
tflite_model = converter.convert()

# 변환된 모델을 .tflite 파일에 저장
open("mnist.tflite", "wb").write(tflite_model)


## Setup the TensorFlow Lite signatures

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

infer = interpreter.get_signature_runner("infer")

### Compare the output of the original model, and the converted lite model:
logits_original = m.infer(x=train_images[:1])["logits"][0]
logits_lite = infer(x=train_images[:1])["logits"][0]

# MNIST 이미지 (1,28,28)
print(train_images[:1].shape)
# 각 숫자들에 대한 inference 결과 (1,10)
print(m.infer(x=train_images[:1]))


### Plot result
def compare_logits(logits):
    width = 0.35
    offset = width / 2
    assert len(logits) == 2

    keys = list(logits.keys())
    plt.bar(
        x=np.arange(len(logits[keys[0]])) - offset,
        height=logits[keys[0]],
        width=0.35,
        label=keys[0],
    )
    plt.bar(
        x=np.arange(len(logits[keys[1]])) + offset,
        height=logits[keys[1]],
        width=0.35,
        label=keys[1],
    )
    plt.legend()
    plt.grid(True)
    plt.ylabel("Logit")
    plt.xlabel("ClassID")

    delta = np.sum(np.abs(logits[keys[0]] - logits[keys[1]]))
    plt.title(f"Total difference: {delta:.3g}")

    plt.show()


compare_logits({"Original": logits_original, "Lite": logits_lite})
