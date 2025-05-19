
==================================================
🎮 OYUN HARİTASI ÜRETİMİ - GAN DESTEKLİ PROJE
==================================================

📌 PROJE AMACI:
Bu proje, kullanıcıdan alınan kısa bir oyun senaryosunu analiz ederek:
1. Senaryoya uygun bir harita planı (metinsel olarak) üretir.
2. DCGAN modeliyle senaryoya uygun yapay bir oyun haritası görseli oluşturur.
3. Flask tabanlı web arayüzü ile hem metinsel hem görsel çıktıyı kullanıcıya sunar.

--------------------------------------------------
🛠 TEKNOLOJİLER:
- Python 3.10+
- PyTorch
- Transformers (Flan-T5)
- Flask
- PIL (Pillow)
- TorchVision

--------------------------------------------------
📁 PROJE DİZİN YAPISI:

project-root/
├── app.py                  # Flask uygulaması
├── generator_model.py      # DCGAN Generator mimarisi
├── generator_infer.py      # Görsel üretim fonksiyonu
├── scenario_enricher.py    # Flan-T5 senaryo zenginleştirme
├── index.html              # HTML frontend
├── static/                 # Üretilen görseller buraya kaydedilir
│   └── generated_*.png
├── map_dataset/            # Eğitim verisi (PNG harita görselleri)
│   └── train/
│       └── *.png
├── generator.pth           # Eğitilmiş model ağırlıkları
└── README.txt              # Bu dosya

--------------------------------------------------
🚀 BAŞLATMA TALİMATLARI:

1. Gerekli paketleri yükle:
   pip install torch torchvision flask transformers pillow

2. Flask uygulamasını çalıştır:
   python app.py

3. Tarayıcıda aç:
   http://127.0.0.1:5000

4. Senaryonu yaz, Enter’a bas. Sistem hem metin hem görsel üretir.

--------------------------------------------------
🔁 DCGAN MODELİNİ YENİDEN EĞİTLEME:

1. Yeni harita görsellerini `map_dataset/train/` klasörüne yerleştir (.png formatında).
2. `dcgan_train.py` eğitim script'ini çalıştır:
   python dcgan_train.py
3. Eğitilen model `generator.pth` olarak kaydedilecektir.
4. Flask uygulaması bu modeli otomatik kullanır.

--------------------------------------------------
🎨 ÖRNEK SENARYO:

A brave explorer descends into an ancient volcano in search of a forgotten artifact.

📤 Üretilen:
- Harita planı: The map is a 3D world with a lava lake. The player controls a character named 'Ana' who is a 'Battlecruiser'. 
The player's goal is to destroy the lava lake and find the artifact.
The player's main enemies are a 'Battlecruiser'. 
The player's main objective is to destroy the lava lake and find the artifact. 

- Harita görseli: static/generated_xxxxxx.png

ÖRNEK ÇIKTI ornek_cıktı.png dosyasıyla verilmiştir.
--------------------------------------------------
👩‍💻 YAZAR: SERHAT TİLEKLİOĞLU
📅 TARİH: Mayıs 2025
📬 E-POSTA: serhattl98@gmail.com
OKUL NO: 220212010


--------------------------------------------------
NOT:
- `.webp` dosyaları desteklenmez, `.png` formatı tercih edilmelidir.
- Görseller `64x64` çözünürlüğünde siyah-beyaz olmalıdır.

==================================================
