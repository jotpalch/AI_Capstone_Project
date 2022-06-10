# AI Capstone Final Group Project

Group Member: **0816044黃則維**, **0816065陳暐誠**, **0816105劉承遠**

## Introduction
圖像風格轉換是從近年以來逐漸熱門的深度學習技術，而其目的是希望生成一個新風格的圖像，並且其圖像有著風格圖片的風格以及內容圖片的內容。
風格轉換屬於紋理問題。傳統上，會透過提取底層的特徵，但不能取得高層的抽象特徵，但在隨著 CNN 的日漸成熟，使得分離內容與風格變得更加容易。

## Related Work
### 生成對抗網路(GAN)
生成對抗網路(GAN)是非監督式學習的一種方法，通過讓兩個神經網路相互博弈的方式進行學習。 生成對抗網路由一個生成網路與一個判別網路組成。生成網路latent space 中隨機取樣作為輸入，其輸出結果需要盡量模仿訓練集中的真實樣本。判別網路的輸入則為真實樣本或生成網路的輸出，其目的是將生成網路的輸出從真實樣本中盡可能分辨出來。而生成網路則要盡可能地欺騙判別網路。兩個網路相互對抗、不斷調整參數，最終目的是使判別網路無法判斷生成網路的輸出結果是否真實。以往訓練神經網路常是透過人類提供大量標記資料供機器分析練習（即監督式學習）比如知名的圍棋人工智慧AlphaGo前期的訓練，便是針對人類輸入的大量棋譜進行監督式學習（後期開始自我對弈的訓練則是非監督式學習，某種層面上也有點對抗訓練的概念）  
![](https://i.imgur.com/tULIiXg.png)

而GAN透過自己相互對抗的生成與鑑別網路，大幅減少資料量的需求，也為非監督式學習提供了更為進步的方法。目前GAN較多被應用在生成資料方面，如圖像與影音的生成、合成、辨識、修復等等，進階一點的則是輸入文本描述便能生成與形容相符的圖像，或者透過語言模型實現機器翻譯等。
### Transfomer

Attention Is All You Need 論文中提出一種全新的神經網路架構命名為 Transformer，是一個利用注意力機制來提高模型訓練速度的模型。Trasnformer可以說是完全基於自注意力機制的一個深度學習模型，因為它適用於並行化計算，和它本身模型的復雜程度導致它在精度和性能上都要高於之前流行的RNN循環神經網絡。  
![](https://i.imgur.com/XbZPYM8.png)
Transformer 的網路架構，由 Encoder-Decoder 堆疊而成，其中 N 為堆疊的層數，預設值為 6。 Encoder-Decoder 裡面的結構，由 Multi-head Attention、Add&Norm、Feed Forward、Masked Multi-head Attention 幾個部分構成。
Self-Attention的實作說明如下：  
![](https://i.imgur.com/YshgYyQ.png)
Transformer 具有較高的計算效率和很好的擴展性，可以支持訓練超過 100 Billion 參數的模型。目前 Transformer 已成為 NLP 領域的主流，衍生出了 BERT、GPT 等模型，但是在計算機視覺領域，Transformer 的應用卻還只是在起步階段。 Google 在 2020 年的一篇論文《An Image is Worth 16*16 Words: Transformers for Image Recognition at Scale》，論文中提出了 VisionTransformer (ViT)，能直接利用 Transformer 對圖像進行分類，而不需要卷積網絡。為了讓 ViT 模型可以處理圖片，首先要把圖片劃分為很多個區塊 (類似 NLP 中的 token)，然後把區塊序列傳入 ViT。  
![](https://i.imgur.com/cSHdpHq.png)
在計算機視覺領域中，多數算法都是保持CNN整體結構不變，在CNN中增加attention模塊或者使用attention模塊替換CNN中的某些部分。有研究者提出，沒有必要總是依賴於CNN。因此，作者提出ViT算法，僅僅使用Transformer結構也能夠在圖像分類任務中表現很好。
受到NLP領域中Transformer成功應用的啟發，ViT算法中嘗試將標准的Transformer結構直接應用於圖像，並對整個圖像分類流程進行最少的修改。具體來講，ViT算法中，會將整幅圖像拆分成小圖像塊，然後把這些小圖像塊的線性嵌入序列作為Transformer的輸入送入網絡，然後使用監督學習的方式進行圖像分類的訓練。
Transformer相較於CNN結構，缺少一定的平移不變性和局部感知性，因此在數據量不充分時，很難達到同等的效果。具體表現為使用中等規模的ImageNet訓練的Transformer會比ResNet在精度上低幾個百分點。
當有大量的訓練樣本時，結果則會發生改變。使用大規模數據集進行預訓練後，再使用遷移學習的方式應用到其他數據集上，可以達到或超越當前的SOTA水平。
## Data Collection
### 爬蟲
我們使用 selenium 套件去動態爬取
https://www.watchshop.com/ 
網站，因為這個手錶的網頁類似 google 圖片的網頁，不會一次載入全部的網站內容，所以如果使用一般的爬蟲技巧，無法一次抓取大量的圖片。
我們爬取了關於自動機械錶以及電子錶有關的圖片。
```
driver = webdriver.Chrome(r'chromedriver') 
driver.get(url)
for i in range(100):
    down += i*500
    js = "document.documentElement.scrollTop=%d" % down 
    driver.execute_script(js)
```
搭配以上這些精簡過的程式碼可以做到打開 Chrome 瀏覽器並動態滑動網頁的效果。
### 網路上下載
部分資料從是從下列連結下載的。
https://images.cv/
以彌補爬蟲資料的不足。

## Preprocessing
我們針對了找來的圖片作了一些預先處理以利丟進模型去訓練。
### 手動刪除圖片
由於資料集是從不同的地方所蒐集而成，因為有些照片較不清楚以及手錶表面沒有完整露出等原因，導致會有一些相片並不是我們要的，或者相片十分模糊，因此我會手動刪除這些照片。
### 調整圖片大小
由於圖片來自不同的資料集，所以資料的大小也不相同，所以我使用 opencv 將每一張圖片調整至大小一樣 ( 400 * 400 ) ，以便於後續模型處理。而我這邊採用 zero padding 的方式進行，將圖片等比例放大，不足的地方則補 0。
```
#### Zero Padding ####

scale = max(h, w) / 64.0
nw, nh = int(w/scale), int(h/scale)
r_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA) 
t, b, l, r = (400-nh)/2+1, (400-nh)/2, (400-nw)/2+1, (400-nw)/2 
zp_img = cv2.copyMakeBorder(r_img, int(t), int(b), int(l), 
         int(r), cv2.BORDER_CONSTANT, value=[0,0,0])
```
### Data Augmentation
因為訓練集的數據量沒有到非常多，所以我使用了 Data Augmentation 的技術擴大我的訓練集，他會水平翻轉抑或是微調照片的角度，使訓練集的數量變為原先的三倍。
```
#### Data Augmentation ####

datagen = ImageDataGenerator(featurewise_center=True, 
                             featurewise_std_normalization=True,
                             rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.1,
                             zoom_range=0.1, horizontal_flip=True,
                             vertical_flip=False, dtype=np.float32)
datagen.flow(array_img, batch_size=2, 
             save_to_dir='aug train diff ratio/ftrain/',
             save_prefix=str(counts)+'_', save_format='jpg')
```
### 去除噪點
由於一些圖片可能含有雜訊，我使用 Gaussian Blur 嘗試將躁點去除，但可能會因此造成圖片變得模糊。
```
#### Gaussian Blur ####

kernel = np.ones((5,5),np.float32)/25
gb_img = cv2.GaussianBlur(img, (3, 3), 0)
```

## Models
### Substitute CNN Model with Transformer in GAN  
![](https://i.imgur.com/jU0siuF.png)

我們嘗試在最初版本的 GAN 上以 Transformer 替換掉原本在  Generator 與 Discriminator 裡面的 CNN 架構。在置換的過程中，光是一開始要將原始的 GAN 跑起來就花了不少工夫，畢竟是一個將近八年前的模型，裡面的架構與程式碼對於現在似乎可以說是有點過時。
在 Discriminator 中， 參考 Vision Transformer 的方式將圖像切割成一塊一塊，並將圖塊依位置標註後丟入 Transformer 中，讓他進行判別真假圖像的訓練。在之後的調整參數以及架構調整上下了好一番功夫，加上參閱好幾篇論文，才讓生成的圖像開始有了手錶的輪廓。
主要概念就是把原本模型中原本含有捲積層的部分，以 Transformer 的架構替換。生成的速度略比原本架構慢一些，但是可以承受的資料量比起 CNN 來的更好。經過實驗，Transformer 架構可以承受比 CNN 架構多30%的資料量。	
### ITTR: Unpair Image-to-Image Translation with Transformers  
![](https://i.imgur.com/I5HcbPH.png)

我們參考此篇 paper 來實作圖片風格轉換。我們的架構是建立在 Contrastive Unpaired Translation (CUT) 之下的，該框架不需要像以往 Cycle-GAN 需要 2 組 GAN 和使用 Cycle-consistency loss 的架構，而是改成對比學習的方式 (Contrastive Learning) 來讓輸出的圖片與輸入的圖片相近。
而我們專注在他所用的生成器 (generator)，他的generator (輸入2張圖片，一張為待轉換之圖片，一張為風格圖片，輸出結果圖片) 是由 3 層 CNN 作為 encoder，之後接上 9 個 Resnet Block (有助於減少 training error)，最後再由 3 層和 encoder 相反的 CNN 作為 decoder 來輸出最終的結果。
而我們參考標題所提到的論文，主要是把中間的 9 個 Resnet Block 改變成 Hybrid Perception Block。因為不論是普通的 CNN 或是具有skip connection 的 ResNet Block，他們都比較能抓到 short-range dependency，換句話來說就是對於小範圍內的特徵比較敏感和了解，但是對於long-range dependency是很薄弱的，為了能同時達到抓到 short 和 long-range dependency，我們使用了 kernel size = 3 X 3，depth-wise 的 CNN 來負責 short-range；Dual Pruned Self-attention來負責long-range，同時降低傳統 multi-head self-attention (MHSA) 的複雜性，最後把這兩項的output做結合來得到所有的特徵。但是因為硬體上面的限制，我們把HPB的數量從原本的9個減少到2個，在盡量以不會降低訓練效果的前提之下實驗。
## Results
以下為利用 GAN 訓練出的圖像結果，由於硬體加上時間的限制，沒辦法來的及產生高畫質的清晰圖片，但在 epoch 較後面所產生的圖像可以看出每一張圖都有明顯的手錶輪廓且風格也有所不同。  
下圖為原始圖像:  
![](https://i.imgur.com/SlA5aRa.png)  
下兩張圖為生成圖像:  
![](https://i.imgur.com/OHvJMR4.png)
![](https://i.imgur.com/YmeZwYi.png)

以下為利用 ITTR 訓練出的圖像結果，因為硬體加上時間的限制，我們沒有辦法訓練太多 epoch。
這是在限制以下我們利用修改過後的 ITTR 所生產出的圖片。左邊是內容圖片，右邊是風格圖片，中間則是生產出的假圖片。
- 手錶  
![](https://i.imgur.com/eYu2zE1.png)
![](https://i.imgur.com/du5tdUe.png)

- 馬 -> 斑馬 : 利用論文裡面所使用的資料集訓練而成  
![](https://i.imgur.com/jDj1FVw.png)
![](https://i.imgur.com/rNWeif0.png)

## Conclusion
由實驗所得，CNN加上Transformer的結構，會比起純CNN 的表現亮眼，這也證明了Transformer 的確可以用在計算機視覺的領域之中。
往常我們都會將 Transformer 與自然語言處理的領域聯想再一起，但除了讓 Transformer 繼續在自然語言處理領域上發光發熱，科學家也致力於將其應用在計算機視覺領域之中，近年來的研究也證實了 Transformer 在圖像領域中的拓展性，期待在未來我們也可以看見其與幾年前的 CNN 一樣在計算機視覺的領域之中佔有一席之地。
## Contribution


| Name| Job | Proportion |
| -------- | -------- | -------- |
| 黃則維     | Research, Coding, Experiment, Report     | 33.3%     |
| 陳暐誠     | Research, Coding, Experiment, Report     | 33.3%     |
| 劉承遠     | Research, Coding, Experiment, Report     | 33.3%     |

## Reference

Creswell, Antonia, et al. "Generative adversarial networks: An overview." IEEE Signal Processing Magazine 35.1 (2018): 53-65.

Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

van den Oord, Aaron, Kalchbrenner, Nal, and Kavukcuoglu, ¨ Koray. Pixel recurrent neural networks. ICML, 2016a. 

Parmar, Niki, et al. "Image transformer." International Conference on Machine Learning. PMLR, 2018.

Jiang, Yifan, Shiyu Chang, and Zhangyang Wang. "Transgan: Two pure transformers can make one strong gan, and that can scale up." Advances in Neural Information Processing Systems 34 (2021).

 Park, T., Efros, A.A., Zhang, R., Zhu, J.Y.: Contrastive learning for unpaired image-to-image translation. In: ECCV’20 (2020)

Zheng, Wanfeng & Li, Qiang & Zhang, Guoxin & Wan, Pengfei & Wang, Zhongyuan. (2022). ITTR: Unpaired Image-to-Image Translation with Transformers. 

Toron, Najiba, Janaina Mourao-Miranda, and John Shawe-Taylor. "TransGAN: a Transductive Adversarial Model for Novelty Detection." arXiv preprint arXiv:2203.15406 (2022).

Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

Zhang, Han, et al. "Self-attention generative adversarial networks." International conference on machine learning. PMLR, 2019.

Mao, Xudong, et al. "Least squares generative adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.

Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.

Miyato, Takeru, et al. "Spectral normalization for generative adversarial networks." arXiv preprint arXiv:1802.05957 (2018).

Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.

Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

Karras, Tero, Samuli Laine, and Timo Aila. "A style-based generator architecture for generative adversarial networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.**
