# Multimodal-Pathological-Voice

多模態病理嗓音分類競賽[競賽連結](https://tbrain.trendmicro.com.tw/Competitions/Details/27)

隊伍名稱: TEAM_3042

## Preprocess

### 病歷資料

參考[preprocess.ipynb](preprocess.ipynb)，其中`Smoking`、`Drinking`、`Onset of dysphonia`、`Diurnal pattern` 種類型欄位以 one-hot 方式處理，其餘數值型欄位以 min-max 方式正規化

### 音訊資料

因音訊長度不一，會將音訊自身頭尾相連的方式 concat 直到長度達最大音訊長度
| Before concat           | After concat            |
| ----------------------- | ----------------------- |
| ![](assets/signal1.png) | ![](assets/signal2.png) |

並引用[BEATs: Audio Pre-Training with Acoustic Tokenizers](https://github.com/microsoft/unilm/tree/master/beats)的[預訓練權重](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)提取音訊特徵，可得到 shape `(408, 768)` 的特徵

## Training

參考[beats-dense.ipynb](beats-dense.ipynb)，模型架構如下圖，將音訊特徵與病歷資料各自經過 `Dense` 後 `concat` 起來做分類
![](assets/model.png)

訓練資料為 90% 資料量，其餘 10% 會在訓練階段以 `accuracy` 評估指標挑選最佳模型權重；訓練階段也會根據類別數量計算對應 `class weight`，提高資料量少的類別權重