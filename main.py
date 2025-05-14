import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def myConv(x, x_zero_index, h, h_zero_index):
    print("Konvolüsyon işlemi başlatılıyor...")
    """
    Hazır kütüphane fonksiyonu kullanmadan 
        [n]=(x∗h)[n]= ∑ x[k]h[n−k] 
       bagıntısı ile iki sinyalin konvolüsyonunu hesaplar.
    
    Parametreler:
      x             : Birinci sinyal (liste veya dizi)
      x_zero_index  : x sinyalinde fiziksel n=0'a denk gelen indeks
      h             : İkinci sinyal (liste veya dizi)
      h_zero_index  : h sinyalinde fiziksel n=0'a denk gelen indeks
    
    Dönüş:
      y             : x ve h'nin konvolüsyonu sonucu elde edilen sinyal (liste)
      y_zero_index  : y sinyalinde fiziksel n=0'a karşılık gelen indeks
    """

    
    output_length = len(x) + len(h) - 1 
    y = [0] * output_length  # Sonuç dizisini sıfırlarla başlatıyoruz.
    
    x_length = len(x)
    h_length = len(h)

    for n in range(output_length):
        acc = 0
        # k, x'indeki indeks; n-k da h'deki indeks
        for k in range(x_length):
            j = n - k
            if 0 <= j < h_length:
                acc += x[k] * h[j]
        y[n] = acc

    
    # Fiziksel zaman eksenindeki n=0'ın y dizisi içindeki indeksi
    # orijinal sinyallerde n=0'ın yer aldığı indeklerin toplamıdır.
    y_zero_index = x_zero_index + h_zero_index
    print("Konvolüsyon işlemi tamamlandı.")
    return y, y_zero_index



def grafiksel_karsilastirma(signal1, start1, title1, signal2, start2, title2, signal3, start3, title3, signal4, start4, title4, save_path='comparison.jpeg', display_duration=5):
    """
    Dört sinyali ayrı alt grafikte (subplot) karşılaştırır. 
    Her sinyalin başlangıç indeksi dışarıdan verilir ve bu indekse göre zaman ekseni hesaplanır.
    Her alt grafik üzerinde sinyale ait üst başlık (title) dışarıdan parametre olarak alınır.
    
    Parametreler:
      signal1, signal2, signal3, signal4 : Karşılaştırılacak sinyaller (liste veya numpy array)
      start1, start2, start3, start4     : Her sinyalin başlangıç indeksi 
                                           (örn. sinyalin fiziksel tanım aralığında n=0'ın hangi indekste olduğunu belirtir)
      title1, title2, title3, title4     : Her sinyalin başlığı
    """
    # Her sinyal için zaman (indis) eksenini oluşturma:
    t1 = [start1 + i for i in range(len(signal1))]
    t2 = [start2 + i for i in range(len(signal2))]
    t3 = [start3 + i for i in range(len(signal3))]
    t4 = [start4 + i for i in range(len(signal4))]
    
    # 4 alt grafik (subplot) oluşturma: Vertical (4x1)
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Sinyal 1
    axs[0].plot(t1, signal1, marker='o', linestyle='-')
    axs[0].set_title("x sinyali")
    axs[0].set_ylabel("Genlik")
    axs[0].grid(True)
    
    # Sinyal 2
    axs[1].plot(t2, signal2, marker='s', linestyle='-')
    axs[1].set_title("h sinyali")
    axs[1].set_ylabel("Genlik")
    axs[1].grid(True)
    
    # Sinyal 3
    axs[2].plot(t3, signal3, marker='^', linestyle='-')
    axs[2].set_title("Konvolüsyon Sonucu")
    axs[2].set_ylabel("Genlik")
    axs[2].grid(True)
    
    # Sinyal 4
    axs[3].plot(t4, signal4, marker='d', linestyle='-')
    axs[3].set_title("Konvolüsyon Sonucu (Numpy)")
    axs[3].set_xlabel("n (Dizin)")
    axs[3].set_ylabel("Genlik")
    axs[3].grid(True)
    
    plt.tight_layout()

    # Grafiği JPEG olarak kaydetme
    plt.savefig(save_path)
    print(f"Grafik '{save_path}' dosyasına kaydedildi.")
    
    # Grafiği belirli bir süre ekranda gösterme
    plt.pause(display_duration)
    plt.close()


def record_audio(duration, filename):
    "belirtilen süre boyunca ses kaydı yapar ve belirtilen dosya adına kaydeder."
    "parametreler : duration: ses kaydının süresi (saniye), filename: kaydedilecek dosya adı"

    #recording_5s.wav
    #recording_10s.wav
    fs = 8000  # Örnekleme frekansı (Hz)
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Kayıt bitene kadar bekleyin
    wav.write(filename, fs, recording)
    print(f"{duration} saniyelik ses kaydı tamamlandı ve '{filename}' dosyasına kaydedildi.")

def play_audio(filename):
    """
    Verilen dosya adındaki WAV ses dosyasını çalar.
    
    Parametreler:
      filename : Çalınacak ses dosyasının adı (string)
    """
    try:
        # WAV dosyasını oku
        fs, data = wav.read(filename)
        
        # Ses dosyasını çal
        print(f"'{filename}' dosyası çalınıyor...")
        sd.play(data, samplerate=fs)
        sd.wait()  # Çalma işlemi bitene kadar bekle
        print(f"'{filename}' dosyası çalma işlemi tamamlandı.")
    except FileNotFoundError:
        print(f"Hata: '{filename}' dosyası bulunamadı.")
    except Exception as e:
        print(f"Hata: Ses dosyası çalınırken bir sorun oluştu: {e}")



def sistem(x, M, A, delay=400):
    """
    Bu fonksiyon, giriş sinyali x[n]'ye aşağıdaki sistemi uygular:
      y[n] = x[n] + ∑ₖ₌₁ᴹ (A · k) · x[n - delay*k]
      
    Parametreler:
      x: Giriş sinyali, numpy dizisi. Boyut: (örnek sayısı, kanal sayısı)
      M: Toplam gecikme katmanı sayısı.
      A: Katsayı parametresi.
      delay: Gecikmenin birim adım değeri (varsayılan 400).
      
    Dönüş:
      y: Sistem çıkışı sinyali, x[n] ile aynı boyutta, int16 formatında numpy dizisi.
    """
    N = x.shape[0]  # Toplam örnek sayısı
    y = x.astype(np.float64).copy()  # İlk terim: x[n]
    
    # Her k için, x[n - delay*k] terimini hesaplayıp y[n]'ye ekliyoruz.
    for k in range(1, M + 1):
        current_delay = delay * k
        shifted = np.zeros_like(y)  # Kaydırılmış sinyal için boş dizi
        
        # Gecikme süresi sinyal uzunluğundan küçükse, kaydırılmış kısım alınır
        if current_delay < N:
            shifted[current_delay:] = x[:N - current_delay]
            
        # A * k ile ağırlıklandırılarak eklenir
        y += A * k * shifted

    # Ses sinyallerinde tip dönüşümü yapılarak taşma (overflow) önlenir
    return np.clip(y, -32768, 32767).astype(np.int16)


def impulse_response(M, A, impulse_length=5000, channels=1, delay=400):
    """
    Bu fonksiyon, sistemimize impuls (delta) sinyali gönderip sistemin 
    dürtü yanıtı h[n]'yi hesaplar.
    
    Parametreler:
      M: Gecikme katmanı sayısı.
      A: Katsayı parametresi.
      impulse_length: Impulse sinyalinin örnek sayısı; sistemin tüm gecikmeleri kapsayacak şekilde belirlenmelidir.
      channels: Kanal sayısı (mono için 1, stereo için 2).
      delay: Gecikme adımı (varsayılan 400).
    
    Dönüş:
      h: Hesaplanan dürtü yanıtı, numpy dizisi.
    """
    # Impulse sinyali oluşturuluyor: İlk örnekte 1, diğer örneklerde 0
    impulse = np.zeros((impulse_length, channels), dtype=np.int16)
    impulse[0, :] = 1
    
    # Giriş olarak impulse verilip sistemin dürtü yanıtı elde ediliyor.
    h = sistem(impulse, M, A, delay)
    return h




def question_one():
    """
    Soru 1: Konvolüsyon işlemi için kullanıcıdan iki sinyal alma ve konvolüsyonu hesaplama.
    """
    # Kullanıcıdan iki sinyal alıyoruz.
    x = list(map(float, input("Birinci sinyali girin (boşlukla ayırın): ").split()))
    h = list(map(float, input("İkinci sinyali girin (boşlukla ayırın): ").split()))

    # Kullanıcıdan her iki sinyalin fiziksel n=0'a karşılık gelen indekslerini alıyoruz.
    x_zero_index = int(input("Birinci sinyal için fiziksel n=0'a karşılık gelen indeksi girin: "))
    h_zero_index = int(input("İkinci sinyal için fiziksel n=0'a karşılık gelen indeksi girin: "))

    # Konvolüsyonu hesaplıyoruz.
    y, y_zero_index = myConv(x, x_zero_index, h, h_zero_index)

    # Sonuçları ekrana yazdırıyoruz.
    print(f"Konvolüsyon sonucu: {y}")
    print(f"Konvolüsyon sonucu sinyalinin fiziksel n=0'a karşılık gelen indeksi: {y_zero_index}")

def question_two():
    """
    Soru 2: konvilsiyon işlemi için kandi yazdığım fonksiyonu ve hazır fonksiyonu  grafiksel ve vektörel karşılaştırma.
    """
    # Kullanıcıdan iki sinyal alıyoruz.
    x = list(map(float, input("Birinci sinyali girin (boşlukla ayırın): ").split()))
    h = list(map(float, input("İkinci sinyali girin (boşlukla ayırın): ").split()))

    # Kullanıcıdan her iki sinyalin fiziksel n=0'a karşılık gelen indekslerini alıyoruz.
    x_zero_index = int(input("Birinci sinyal için fiziksel n=0'a karşılık gelen indeksi girin: "))
    h_zero_index = int(input("İkinci sinyal için fiziksel n=0'a karşılık gelen indeksi girin: "))

    # Konvolüsyonu hesaplıyoruz.
    y, y_zero_index = myConv(x, x_zero_index, h, h_zero_index)

    # Numpy'nin hazır convolve fonksiyonunu kullanarak konvolüsyonu hesaplıyoruz.
    y_numpy = np.convolve(x, h)

    y_numpy_zero_index = x_zero_index + h_zero_index
    #sonuclar icin grafiksel karsilastirma
    grafiksel_karsilastirma(x, x_zero_index, "x", h, h_zero_index,"h", y, y_zero_index,"myY", y_numpy, y_numpy_zero_index,"numpyY", save_path='ayrik_konvolusyon_karsilastirma.jpeg')
    # Sonuçları ekrana yazdırıyoruz.
    print(f"x sinyali: {x} , fiziksel n=0'a karşılık gelen indeks: {x_zero_index}")
    print(f"y sinyali: {h}, fiziksel n=0'a karşılık gelen indeks: {h_zero_index}")
    print(f"Konvolüsyon sonucu: {y}, fiziksel n=0'a karşılık gelen indeks: {y_zero_index}")
    print(f"Konvolüsyon sonucu (Numpy): {y_numpy}, fiziksel n=0'a karşılık gelen indeks: {y_numpy_zero_index}")

def question_three():
    """
    Soru 3: Konvolüsyon işleminde kullanılmak için 5 ve 10 saniyelik ses kaydetme (tek kanallı).
    """
    # 5 saniyelik ses kaydı yapma
    print("5 saniyelik ses kaydını başlatmak için 'Enter' tuşuna basın.")
    input()  # Kullanıcıdan 'Enter' tuşuna basmasını bekliyoruz.
    record_audio(5, "recording_5s_mono.wav")
    print("5 saniyelik ses kaydı yapıldı ve 'recording_5s_mono.wav' dosyasına kaydedildi.")
    
    # 10 saniyelik ses kaydı yapma
    print("10 saniyelik ses kaydını başlatmak için 'Enter' tuşuna basın.")
    input()  # Kullanıcıdan 'Enter' tuşuna basmasını bekliyoruz.
    record_audio(10, "recording_10s_mono.wav")
    print("10 saniyelik ses kaydı yapıldı ve 'recording_10s_mono.wav' dosyasına kaydedildi.")

def question_four():
    """
    Soru 4: kaydedilen seskaydini farkli M değerleri ile işleme.
    """

    M=[3, 4, 5]  # Gecikme katmanı sayıları
    A=0.5  # Katsayı parametresi        

    #x1 ve x2 ses kaydı icin farklı M değerleri ile işleme
    fs1, x1 = wav.read("recording_5s_mono.wav") # 5 saniyelik ses kaydını oku  
    fs2, x2 = wav.read("recording_10s_mono.wav") # 10 saniyelik ses kaydını oku

    #m degerleri icin işleme döngüsü
    for m in M:
        # 5 saniyelik ses kaydını işleme
        # 5 saniyelik ses kaydını işleme
        h1 = impulse_response(m, A, impulse_length=5000, channels=1)
        y1_5, y1_5_zero_index = myConv(x1.flatten(), 0, h1.flatten(), 0)  # Konvolüsyon işlemi
        wav.write(f"myConv_M{m}_5s.wav", fs1, np.array(y1_5, dtype=np.int16))
        print(f"5 saniyelik ses kaydı için myConv ile M={m} ile işleme yapıldı ve 'myConv_M{m}_5s.wav' dosyasına kaydedildi.")
        
        y2_5 = np.convolve(x1.flatten(), h1.flatten(), mode='full')
        wav.write(f"numpyConv_M{m}_5s.wav", fs1, np.array(y2_5, dtype=np.int16))

        print(f"5 saniyelik ses kaydı için Numpy Conv ile M={m} ile işleme yapıldı ve 'numpyConv_M{m}_5s.wav' dosyasına kaydedildi.")
        
        # Gecikmiş iki sinyali karşılaştırma
        grafiksel_karsilastirma(x1.flatten(), 0,"x", h1.flatten(), 0,"h", y1_5, y1_5_zero_index,"myY" ,y2_5, 0,"numpyY", save_path=f'karsilastirma_5s_M{m}.jpeg')

        # 10 saniyelik ses kaydını işleme
        h2 = impulse_response(m, A, impulse_length=5000, channels=1)
        y1_10, y1_10_zero_index = myConv(x2.flatten(), 0, h2.flatten(), 0)
        wav.write(f"myConv_M{m}_10s.wav", fs2, np.array(y1_10, dtype=np.int16))
        print(f"10 saniyelik ses kaydı için myConv ile M={m} ile işleme yapıldı ve 'myConv_M{m}_10s.wav' dosyasına kaydedildi.")
        
        y2_10 = np.convolve(x2.flatten(), h2.flatten(), mode='full')
        wav.write(f"numpyConv_M{m}_10s.wav", fs2, np.array(y2_10, dtype=np.int16))
        print(f"10 saniyelik ses kaydı için Numpy Conv ile M={m} ile işleme yapıldı ve 'numpyConv_M{m}_10s.wav' dosyasına kaydedildi.")
        
        # Gecikmiş iki sinyali karşılaştırma
        grafiksel_karsilastirma(x2.flatten(), 0, "x",h2.flatten(), 0, "h",y1_10, y1_10_zero_index,"myY", y2_10, 0,"numpyY", save_path=f'karsilastirma_10s_M{m}.jpeg')
    
    # Ses dosyalarını çalma
    print("5 saniyelik ses kaydının tüm versiyonlarını (her biri icin once myConv sonra numpy sonuclarını) çalmak için 'Enter' tuşuna basın.")

    input()
    play_audio("myConv_M3_5s.wav")
    play_audio("numpyConv_M3_5s.wav")
    play_audio("myConv_M4_5s.wav")
    play_audio("numpyConv_M4_5s.wav")
    play_audio("myConv_M5_5s.wav")
    play_audio("numpyConv_M5_5s.wav")

    print("10 saniyelik ses kaydının tüm versiyonlarını (her biri icin once myConv sonra numpy sonuclarını) çalmak için 'Enter' tuşuna basın.")
    input() 

    play_audio("myConv_M3_10s.wav")
    play_audio("numpyConv_M3_10s.wav")
    play_audio("myConv_M4_10s.wav")
    play_audio("numpyConv_M4_10s.wav")
    play_audio("myConv_M5_10s.wav")
    play_audio("numpyConv_M5_10s.wav")
    



   
    
   
#main fonksiyonu
def main():
    choice = '' 
    """
    Ana fonksiyon: Kullanıcıdan hangi sorunun çalıştırılacağını seçmesini ister.
    """
    print("Lütfen çalıştırmak istediğiniz soruyu seçin:")
    print("1. Soru 1")
    print("2. Soru 2")
    print("3. Soru 3")
    print("4. Soru 4")
    
    
    while (choice != 'q' ):
        choice = input("Seçiminizi yapın (1/2/3/4) çikmak için q ya basin: ")
        if choice == '1':
            question_one()
        elif choice == '2':
            question_two()
        elif choice == '3':
            print("yeni ses kaydı yaparsanız eski kaydı kaybedeceksiniz. Devam etmek için 'Enter' tuşuna basın. ")
            input()
            question_three()
        elif choice == '4':
            question_four()
        elif choice == 'q':
            print("Çıkış yapılıyor...")
            return
        else:
            print("Geçersiz seçim!")

main()
#main fonksiyonu çağrıldı ve program başlatıldı.
# Bu kod, konvolüsyon işlemi, ses kaydı ve işleme ile ilgili temel işlevleri içermektedir.
