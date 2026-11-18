# Konvolüsyon İşleminin Kodlanması

## Amaç ve kapsam

Bu çalışmada konvolüsyon, max pooling, ReLU ve softmax işlemleri ile Fashion-MNIST üzerinde küçük bir CNN incelenmiştir. Kod, başlangıç filtrelerini görselleştirir, verilen örnek işlemleri çalıştırır ve aynı eğitim ayarlarıyla ilk katmanda 8 ve 16 filtre kullanılan iki modeli eğitir. Eğitim metrikleri donanıma ve TensorFlow sürümüne göre değişebildiğinden, deney tablosu kod çalıştırıldığında gerçek ölçümlerle oluşturulmalıdır.

## Konvolüsyon katmanı

İlk örnekte 16 filtre, katmanın 16 ayrı özellik haritası üretmesi demektir. Eğitim sırasında bu filtreler girişteki farklı örüntüleri, örneğin kenar veya basit dokuları, yakalamayı öğrenir. Tek kanallı girdi ve 3×3 çekirdek için her filtre 3×3×1 ağırlıktan oluşur.

`weights` tensörünün biçimi `(3, 3, 1, 16)` olur: çekirdek yüksekliği, çekirdek genişliği, giriş kanalı ve filtre sayısı. `biases` biçimi `(16,)` olur; her filtrenin kendi bias değeri vardır.

Girdi `(1, 128, 128, 1)` biçimindedir: batch, yükseklik, genişlik, kanal. Padding belirtilmediğinden TensorFlow `valid` kullanır; stride 1 ile çıktı uzamsal boyutu `(128−3)/1+1 = 126` olur. Dolayısıyla çıktı `(1, 126, 126, 16)` biçimindedir. `he_normal`, ReLU gibi aktivasyonlarda başlangıç sinyalinin ölçeğini korumaya yönelik He normal dağılımından başlangıç ağırlıkları çeker. Rastgele başlangıç ağırlıklarıdır; eğitimle güncellenir.

Kod 16 başlangıç çekirdeğini `konvolusyon_filtreleri.png` adıyla kaydeder. İlk örnekteki 16 filtre, eğitim öncesi rastgele başlangıç değerleridir; anlamlı görsel örüntüler öğrenilmiş filtrelerle karıştırılmamalıdır.

## Max Pooling

Girdi 4×4 matrisinde, stride 2 olan 2×2 pencereler çakışmadan taranır:

| Bölge | En büyük değer |
|---|---:|
| Sol üst: `[[1, 3], [5, 6]]` | 6 |
| Sağ üst: `[[2, 4], [1, 2]]` | 4 |
| Sol alt: `[[7, 2], [4, 1]]` | 7 |
| Sağ alt: `[[8, 3], [5, 9]]` | 9 |

Sonuç `[[6, 4], [7, 9]]` olur. Girdi `(1, 4, 4, 1)`, çıktı `(1, 2, 2, 1)` biçimindedir. Pooling, en belirgin aktivasyonu koruyup uzamsal boyutu küçültür. Bu, sonraki hesaplamaları azaltır ve küçük konum değişimlerine karşı dayanıklılık sağlayabilir.

## ReLU ve softmax

ReLU fonksiyonu `max(0, x)` biçimindedir. Negatif girdileri sıfıra eşler; pozitifleri aynen bırakır. Böylece ağa doğrusal olmayan bir dönüşüm kazandırır. ReLU olmasaydı ardışık doğrusal katmanlar doğrusal kalır ve model karmaşık karar sınırlarını öğrenmekte zorlanırdı.

Softmax sınıf skorlarını toplamı 1 olan değerlere dönüştürür. `[2.0, 1.0, 0.1]` için yaklaşık çıktı `[0.659, 0.242, 0.099]` olur. Bunlar modelin üç sınıfa atadığı normalize edilmiş skor/olasılık yorumudur. En yüksek değerin indeksi `argmax` ile seçilir; burada indeks 0'dır. Softmax değerleri tek başına modelin doğru olduğuna dair garanti vermez.

## Fashion-MNIST modeli

Görüntüler 28×28 gri seviyelidir. Piksel değerleri 255'e bölünerek 0–1 aralığına getirilir ve son eksene kanal boyutu eklenir. Eğitim verisi `(60000, 28, 28, 1)`, test verisi `(10000, 28, 28, 1)` biçimindedir.

Mimari iki konvolüsyon ve iki max pooling katmanına sahiptir. İlk katmanda 8 filtre daha basit özellikleri, ikinci katmanda 16 filtre önceki katmanın özelliklerinden daha karmaşık birleşimleri öğrenebilir. Filtre sayısını artırmak temsil kapasitesini ve hesaplama maliyetini artırır; başarının mutlaka artacağı anlamına gelmez.

| Aşama | Uzamsal boyut | Kanal/özellik sayısı |
|---|---:|---:|
| Girdi | 28×28 | 1 |
| Conv2D, 8 filtre, valid | 26×26 | 8 |
| MaxPooling2D, 2×2 | 13×13 | 8 |
| Conv2D, 16 filtre, valid | 11×11 | 16 |
| MaxPooling2D, 2×2 | 5×5 | 16 |
| Flatten | — | 5×5×16 = 400 |
| Dense | — | 64 nöron |
| Çıkış Dense | — | 10 sınıf |

Flatten, üç boyutlu özellik haritasını 400 elemanlı vektöre çevirir. Dense katmanındaki 64, öğrenilebilir nöron sayısıdır. Çıkışta 10 nöron vardır; Fashion-MNIST'in 10 sınıfına karşılık gelir. Çıkıştaki en yüksek softmax değeri modelin tahmin ettiği sınıfı gösterir. Model başarı metriği olarak accuracy (doğruluk) kullanır; kayıp fonksiyonu sparse categorical crossentropy'dir.

Sınıf indeksleri: 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot.

## Filtre sayısı deneyi

İki deneyde epoch sayısı (5), batch size (64), optimizer (Adam), veri bölünmesi ve diğer katmanlar sabittir; ilk konvolüsyon filtresi sayısı 8'den 16'ya çıkarılır. Betik eğitim doğruluğunu son epoch'tan, test doğruluğunu değerlendirmeden ve eğitim süresini yalnızca `fit` süresinden alıp `deney_sonuclari.csv` dosyasına yazar. Grafikler eğitim ve doğrulama doğruluğu/kaybını karşılaştırır.

| İlk katman filtre sayısı | Son epoch eğitim doğruluğu | Test doğruluğu | Eğitim süresi (sn) |
|---:|---:|---:|---:|
| 8 | Betik çalıştırılınca ölçülür | Betik çalıştırılınca ölçülür | Betik çalıştırılınca ölçülür |
| 16 | Betik çalıştırılınca ölçülür | Betik çalıştırılınca ölçülür | Betik çalıştırılınca ölçülür |

Sonuç değerlendirmesinde daha yüksek test doğruluğuna ek olarak eğitim-doğrulama farkı ve süre de karşılaştırılmalıdır. Eğitim doğruluğu artarken doğrulama/test doğruluğu geriliyorsa aşırı öğrenme olasılığı vardır. Filtre sayısının iki katına çıkması ilk katmandaki parametre sayısını artırır; daha çok hesaplama gerektirir ve veri/model koşullarına göre sonuç kötüleşebilir.

## Çalıştırma ve teslim

Terminalde gerekli paketler kurulduktan sonra `python cnn_odev.py` komutu çalıştırılır. Betik konsola boyutları ve deney sonuçlarını yazar; filtre görselini `konvolusyon_filtreleri.png`, grafiklerini `egitim_grafikleri.png`, ölçüm tablosunu `deney_sonuclari.csv` olarak çalışma dizinine kaydeder. Bu ortamda TensorFlow yüklü olmadığından bu rapora gerçek eğitim ölçümü eklenmemiştir; kod TensorFlow bulunan ortamda çalıştırılınca tablo ve grafikler tamamlanır.

## Sonuç

Konvolüsyon filtreleri öğrenilebilir özellik çıkarıcılar, pooling boyut azaltıcı ve özellik koruyucu, ReLU doğrusal olmayanlık sağlayıcı, softmax ise sınıf skorlarını normalize eden çıkış fonksiyonudur. Fashion-MNIST mimarisindeki uzamsal boyutlar valid konvolüsyon ve 2×2 pooling işlemlerinin ardışık etkisiyle 28×28'den 5×5'e iner. Daha fazla filtre daha zengin özellik temsili sunabilir; ancak doğruluk, genelleme ve süre ölçümleri birlikte değerlendirilmelidir.
