# Reddit Recovery Guide — "Postum Paylaşır Paylaşılmaz Siliniyor"

> **Durum:** Yeni hesap, ilk postlar r/Python'da paylaşır paylaşınmaz otomatik
> kaldırılıyor. Mod eliyle değil, **Reddit AutoMod filtresi** tarafından.
> Bu rehber, o filtreyi aşmak için 3 haftalık kurtarma planıdır.

---

## Teşhis: Neden siliniyor?

Sıralı olasılıklar (en yüksekten en düşüğe):

1. **Yeni hesap filtresi (~%85)** — Hesap <30 gün, karma <50. Reddit
   otomatik "spam olasılığı yüksek" işaretleyip modqueue'a atar. Modlar
   bakmazsa post yayınlanmamış gibi görünür.
2. **Self-promo pattern (~%10)** — Comment geçmişi yok, ilk post self-link.
   AutoMod "sadece kendi içeriğini paylaşan hesap" filtresine takılır.
3. **Subreddit kuralı (~%5)** — r/Python link-post değil text-post bekler,
   veya sadece "Show & Tell Tuesday" thread'inde kabul eder.

---

## Kesin teşhis (2 dakika)

Bir kaldırılan postun URL'si ile:

1. `https://reveddit.com/` aç → URL'yi yapıştır.
   - **"removed by moderators"** → mod kuralı ihlali.
   - **"removed by Reddit"** → spam filtresi (yeni hesap).
   - **"visible"** → aslında görünüyordur, traction problemi.
2. Gizli pencerede (login olmadan) post URL'sini aç.
   - **"not found"** → kaldırılmış, teşhis doğrudur.
   - Görünüyorsa → trafik problemi, bu rehber senin durumun değil.

---

## Çözüm: 3 haftalık kurtarma planı

### Hafta 1 — Hesap warm-up (sıfır self-promo)

**Amaç:** Reddit'e "gerçek kullanıcı" sinyali ver. Hiç kendi linkini paylaşma.

- [ ] Her gün **2-3 cevap yaz** r/Python, r/learnpython, r/programming thread'lerinde.
- [ ] Cevaplar **faydalı** olsun: birinin NumPy sorusuna cevap, bir kod önerisi, bir kaynak linki (kendi projen değil).
- [ ] Upvote alabilmek için **kaliteli cevaplar** — "katılıyorum" değil, gerçek teknik içerik.
- [ ] **Kendi projeni HİÇ linkleme** bu hafta. Sıfır.
- [ ] Hedef: hafta sonunda ~30-50 comment karma.

### Hafta 2 — Doğru subreddit seçimi + draft hazırlığı

Senin projen için **en yüksek dönüşüm** getirecek subreddit'ler:

| Subreddit | Neden işe yarar | Risk |
|-----------|-----------------|------|
| **r/learnpython** | "Sıfırdan FFT/RK4 nasıl yazılır" angle'ı, yeni başlayanlara hitap eder, self-promo toleransı yüksek | Düşük |
| **r/Python** (Show & Tell Salı) | Sadece Salı thread'inde kabul eder, stand-alone post agresif moderasyona takılır | Orta |
| **r/Python (stand-alone)** | En yüksek trafik ama en katı moderasyon | Yüksek |
| **r/dataisbeautiful** | ASCII visualization / dashboard angle'ı | Orta |

**Senin için en güvenli ilk deneme: r/learnpython**, "öğretici kaynak paylaşıyorum" angle'ı ile.

### Hafta 3 — İlk gerçek paylaşım (warm-up sonrası)

Kontroller:

- [ ] Hesap en az **30 günlük** mü? (Reddit yaş filtresi için kritik eşik)
- [ ] Comment karma **50+** mı?
- [ ] Hedef subreddit'te son 1 haftada **en az 5 cevap** vermiş misin?
- [ ] Post **text-post** mu (link-post değil)? Link-postlar yeni hesaplarda daha kolay silinir.
- [ ] **Postu paylaşmadan 1 saat önce** o subreddit'in son postlarına cevap vermiş misin?

Hepsi ✓ ise paylaş.

---

## Postu paylaşır paylaşmaz yapılacaklar

1. **İlk 30 dakika kritik.** Postu paylaş, tarayıcıyı kapatma.
2. **5 dakika sonra refresh et.** Hâlâ visible mı?
   - Görünüyorsa → yorumlara cevap vermeye başla.
   - Kaybolduysa → hemen modlara mesaj at:
     > "Hi mods, my post seems to have been auto-removed. I'm new to the
     > community — happy to follow any rules I missed. Could you check?
     > [post URL]"

3. **Modlar cevap verirse** kuralları uygula, tekrar dene.
4. **Cevap vermezlerse** o subreddit'i bırak, hafta 1-2'ye geri dön.

---

## Asla yapma

- ❌ **Aynı postu tekrar paylaşma.** Reddit bunu "spamPersistent" işaretler,
  hesabın shadowban yer. 7 gün bekle, farklı angle'la dene.
- ❌ **Birden fazla subreddit'e aynı anda cross-post.** Yeni hesap için
  otomatik ban sebebi.
- ❌ **"Star verin" / "destekleyin"** dilini kullanma. Hem modlar hem
  algoritma bunu cezalandırır.
- ❌ **VPN ile yeni hesap açıp aynı linki paylaşma.** Reddit fingerprint
  ile yakalıyor, iki hesap da ban yer.

---

## Alternatif: Reddit'i bekle, paralel kanalları aç

Reddit warm-up sürerken (3 hafta), şu kanallar **hesap yaşı filtresine takılmaz**:

| Kanal | Hesap gereksinimi | İlk trafik |
|-------|-------------------|-----------|
| **dev.to** | Yok | 1-7 gün |
| **Hacker News (Show HN)** | Yok, ama içerik kalitesi çok yüksek olmalı | Anında veya hiç |
| **awesome-* GitHub listeleri** | GitHub hesabı yeterli | 1-4 hafta (PR merge sonrası) |
| **X / Twitter thread** | Yok | 1-3 gün |

Bu kanallardan en az biri açıkken Reddit'i de denemek mantıklı — tek kanala
bağımlı kalmak (Reddit gibi hesap yaşı filtresi olan bir kanala) riskli.

---

## Karar verme

Eğer bu planı uygulamak **3 hafta** sürüyor ve bu süre zarmanı tek bir kullanıcı
bile kazanamıyorsan → **Reddit'i askıya al, dev.to + awesome listelere
odaklan.** Reddit yeni hesapla düşük ROI'li bir kanal; bu senin suçun değil,
platform politikası.

---

## TL;DR

- Postların yeni hesap olduğu için siliniyor, içerik kalitesinden değil.
- 3 haftalık warm-up: her gün 2-3 cevap, sıfır self-link.
- Hafta 3'te r/learnpython'da text-post dene (r/Python değil).
- Bu arada dev.to + awesome listeler paralel açık.
- Tekrar tekrar paylaşma → shadowban.
