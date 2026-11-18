"""Konvolüsyon, pooling, aktivasyon ve Fashion-MNIST CNN ödevi.

Kurulum: python -m pip install tensorflow matplotlib numpy
Çalıştırma: python cnn_odev.py
"""
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 42
tf.keras.utils.set_random_seed(SEED)


def convolution_demo():
    conv = tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation="relu",
        kernel_initializer="he_normal"
    )
    x = tf.random.normal((1, 128, 128, 1))
    y = conv(x)
    weights, biases = conv.get_weights()
    print("Giriş boyutu:", x.shape)
    print("Çıkış boyutu:", y.shape)
    print("Filtre ağırlıklarının boyutu:", weights.shape)
    print("Bias boyutu:", biases.shape)
    print("İlk filtrenin ağırlıkları:\n", weights[:, :, 0, 0])

    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        ax.imshow(weights[:, :, 0, i], cmap="coolwarm", interpolation="nearest")
        ax.set_title(f"Filtre {i + 1}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Başlangıç konvolüsyon filtreleri (3×3)")
    fig.tight_layout()
    fig.savefig("konvolusyon_filtreleri.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def pooling_demo():
    x = tf.constant([[[[1.], [3.], [2.], [4.]],
                      [[5.], [6.], [1.], [2.]],
                      [[7.], [2.], [8.], [3.]],
                      [[4.], [1.], [5.], [9.]]]])
    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    y = pool(x)
    print("Pooling girişi:\n", x.numpy()[0, :, :, 0])
    print("Pooling sonucu:\n", y.numpy()[0, :, :, 0])
    print("Boyutlar:", x.shape, "->", y.shape)


def activation_demo():
    x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
    print("ReLU:", tf.keras.layers.ReLU()(x).numpy())
    logits = tf.constant([[2.0, 1.0, 0.1]])
    probs = tf.keras.layers.Softmax()(logits)
    print("Softmax:", probs.numpy())
    print("Toplam:", tf.reduce_sum(probs).numpy())
    print("En olası sınıf indeksi:", tf.argmax(probs, axis=1).numpy())


def make_model(first_filters=8):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(first_filters, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


def train_experiment(first_filters, x_train, y_train, x_test, y_test):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = make_model(first_filters)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    start = time.perf_counter()
    history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                        validation_split=0.1, verbose=2)
    elapsed = time.perf_counter() - start
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    result = {
        "ilk_filtre": first_filters,
        "egitim_dogrulugu": float(history.history["accuracy"][-1]),
        "test_dogrulugu": float(test_accuracy),
        "egitim_suresi_sn": elapsed,
        "test_loss": float(test_loss),
    }
    return model, history, result


def main():
    convolution_demo()
    pooling_demo()
    activation_demo()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    print("Eğitim verisi:", x_train.shape, "Test verisi:", x_test.shape)

    results = []
    histories = []
    for count in (8, 16):
        model, history, result = train_experiment(count, x_train, y_train, x_test, y_test)
        print(f"\nİlk katmanda {count} filtre - model özeti")
        model.summary()
        print("Test loss:", result["test_loss"], "Test accuracy:", result["test_dogrulugu"])
        results.append(result); histories.append((count, history))

    print("\nDeney karşılaştırması")
    print("İlk filtre | Eğitim doğruluğu | Test doğruluğu | Eğitim süresi (sn)")
    for r in results:
        print(f"{r['ilk_filtre']:>10} | {r['egitim_dogrulugu']:.4f} | "
              f"{r['test_dogrulugu']:.4f} | {r['egitim_suresi_sn']:.2f}")
    np.savetxt("deney_sonuclari.csv",
               np.array([[r["ilk_filtre"], r["egitim_dogrulugu"],
                          r["test_dogrulugu"], r["egitim_suresi_sn"]] for r in results]),
               delimiter=",", header="ilk_filtre,egitim_dogrulugu,test_dogrulugu,egitim_suresi_sn",
               comments="", fmt=["%d", "%.6f", "%.6f", "%.3f"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for count, hist in histories:
        axes[0, 0].plot(hist.history["accuracy"], label=f"{count} filtre")
        axes[0, 1].plot(hist.history["val_accuracy"], label=f"{count} filtre")
        axes[1, 0].plot(hist.history["loss"], label=f"{count} filtre")
        axes[1, 1].plot(hist.history["val_loss"], label=f"{count} filtre")
    for ax, title in zip(axes.flat, ["Eğitim doğruluğu", "Doğrulama doğruluğu",
                                     "Eğitim kaybı", "Doğrulama kaybı"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig("egitim_grafikleri.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
