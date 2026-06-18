# GAN Anime Face Generator (PyTorch)

Implementasi **Generative Adversarial Network (GAN)** menggunakan PyTorch untuk menghasilkan citra wajah anime dari noise acak.

---

## 📌 Deskripsi

Project ini mengimplementasikan model **GAN (Generative Adversarial Network)** yang terdiri dari dua jaringan utama:

- **Generator**: menghasilkan citra anime dari vektor noise acak
- **Discriminator**: membedakan citra asli dan citra hasil generator

Model dilatih secara adversarial hingga Generator mampu menghasilkan gambar yang menyerupai data asli.

Dataset yang digunakan adalah **Anime Face Dataset** dari Kaggle (~63.000 gambar wajah anime).

---

## 🧠 Arsitektur Model

### Generator
- Input: noise vector (100 dimensi)
- Fully Connected Layer
- Transposed Convolution Layers
- Batch Normalization
- ReLU Activation
- Tanh Activation (output image)

### Discriminator
- Convolutional Neural Network (CNN)
- LeakyReLU Activation
- Batch Normalization
- Fully Connected Layer
- Sigmoid Activation (real/fake classification)

---

## 📂 Struktur Project

```text
project/
│
├── gan.py              # Script utama untuk training model GAN
│
└── anime/              # Dataset gambar
    └── faces/          # Kumpulan gambar wajah anime untuk training
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```
---

## ⚙️ Instalasi

Clone repository ini:

```bash
git clone https://github.com/yumairai/GAN_Image_Generation.git
cd gan-anime-generator
