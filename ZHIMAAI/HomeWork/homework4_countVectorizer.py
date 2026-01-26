from pathlib import Path
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
import matplotlib.font_manager as fm
import re

class TxtPreProcess:
    def __init__(self,url):
        self.url = url

    def read_file(self):
        with open(self.url, 'r', encoding='utf-8') as file:
             content = file.read()
        return content 
     
    def chinese_word_tokenize(self,origian_text):
        # 中文用jieba分词，去除标点符号，再传入以空格分隔的字符串
        ##匹配任何不是单词字符\w 或空白字符\s 的字符，即任何标点符号
        pattern = re.compile(r'[^\w\s]')
        words = [word for word in jieba.lcut(origian_text) if not pattern.search(word)]
        return ' '.join(words)


## 预处理
SCRIPT_DIR = Path(__file__).parent
txt_path = SCRIPT_DIR / '碧血剑.txt'
txtCV = TxtPreProcess(txt_path)
# print(text_content)
txt_segmented = txtCV.chinese_word_tokenize(txtCV.read_file())
# print (txt_segmented.split(" "))

## binary=True 表示返回稀疏矩阵, one-hot 编码，出现和没有出现，没有词频统计
##vectorizer = CountVectorizer(binary=True,analyzer='char', ngram_range=(1, 3)) 


vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3)) 
## 分词结果以空格连接
X = vectorizer.fit_transform(txt_segmented.split(" "))
print(X.toarray())
#获取词表和总词频取词表和总词频
feature_names = vectorizer.get_feature_names_out()
# print(feature_names)
word_freq = X.sum(axis=0).A1  # 转为一维数组
# print(word_freq)

df = pd.DataFrame({'word': feature_names, 'freq': word_freq})
# print(df)
df = df.sort_values(by='freq', ascending=False).reset_index(drop=True)
top20_df = df.head(20)
# print(top20_df)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(top20_df['word'],top20_df['freq'], color='skyblue')
ax[0].set_title("Top 20 words frequcy visilize")
ax[0].tick_params(axis='x', rotation=45)
ax[0].tick_params(axis='x', labelsize=10)
font_path=r"C:\Windows\Fonts\msyh.ttc"  
font_prop = fm.FontProperties(fname=font_path)

for lbl in ax[0].get_xticklabels():
    lbl.set_fontproperties(font_prop)

client = wordcloud.WordCloud(font_path=font_path,)
client.generate_from_text(txt_segmented).to_image()
ax[1].imshow(client)
ax[1].axis('off')
ax[1].set_title("Top 20 words cloud")

plt.show()