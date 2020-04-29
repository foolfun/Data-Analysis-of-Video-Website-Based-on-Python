# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 20:40:33 2018

@author: zsl
"""
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from PIL import Image
import jieba
from wordcloud import WordCloud as wc,ImageColorGenerator
#1数据探索
df = pd.read_csv('d:/bili2.csv')
print(df.info())
#1.1默认统计数值型数据每列数据平均值，标准差，最大值，最小值，25%，50%，75%比例。
print(df.describe())
#1.2数据行数和列数
print("最初", df.shape)
#1.3查看数据
print(df.head())
print(df.sample(2))

#2数据处理
#2.1筛选时间属性，将时间属性设为索引
df.set_index('发布时间',inplace=True)
df=df.sort_index()
df['总天数']=pd.to_datetime(df['采集时间'])-pd.to_datetime(df.index)

#2.2删除不需要的列
df=df.drop(['粉丝数','视频HTML地址','视频介绍',
            '发布者头像链接','页面网址','发布者','采集时间','投稿数'],axis=1)

#再次查看数据情况
print(df.info())

#2.3缺失值和缺失值处理,并将所有空值和含有nan项的row删除
df['总播放数'].isnull().value_counts()
df=df.dropna()

#2.4查找重复值，发现没有
sum(df.duplicated())
#将总播放数和总弹幕数、分享数改成float64
print(df['总播放数'].describe())
def _change_type(y1,y2):
    count=0   
    for i in df[y1]:
        if(re.search('\d*(\.\d*)?万',i)):
            x=i.replace('万','0')
            x=float(x)
            x=10000*x
            df.iloc[count,y2]=str(x)
        count=count+1
    print(count)
_change_type('分享数',6)
_change_type('总弹幕数',3)
_change_type('总播放数',2)
df['总播放数']=df['总播放数'].astype('float64')
df['总弹幕数']=df['总弹幕数'].astype('float64')
df['分享数']=df['分享数'].astype('float64')

#2.5总播放数和总弹幕数、硬币、收藏数、分享数做相关性分析
tmpdf=df[['分享数','收藏数','硬币','总弹幕数','总播放数']]
tmpdf=tmpdf.rename(columns={'分享数':'shares',
                            '收藏数':'stores',
                            '硬币':'bs',
                            '总弹幕数':'comments',
                            '总播放数':'plays'})
dfcorr=tmpdf.corr()
# 设置画面大小
plt.subplots(figsize=(9, 9)) 
sns.heatmap(dfcorr, annot=True, vmax=1, square=True, 
            cmap="Blues")
plt.savefig('d:/picture/heatmap.png')
plt.show()
#筛选 合并
df=df.drop(['分享数','收藏数'],axis=1)
tmpdf=tmpdf.drop(['shares','stores'],axis=1)
count=0
for i in df['总天数']:
    #print(i.days)
    x=i.days
    df.iloc[count,6]=str(x)
    count=count+1
print(count)
df['总天数']=df['总天数'].astype('float64')
#数据标准化
tmpdf['days']=df['总天数']
df['hot_index']=(tmpdf['comments']+tmpdf['bs']+tmpdf['plays'])/(tmpdf['days'])
df=df[~df['hot_index'].isin(['inf'])]
df['hot_index']=(df['hot_index']-np.min(df['hot_index']))/(np.max(df['hot_index'])-np.min(df['hot_index']))
#3绘图
#3.1绘制饼图
#查找含有某标签的行
#学科
a1=sum(df['视频标签'].str.contains('数学'))
a2=sum(df['视频标签'].str.contains('英语'))
a3=sum(df['视频标签'].str.contains('政治'))
a4=sum(df['视频标签'].str.contains('专业课'))
print(a1,a2,a3,a4)
lab1='math','english','politce','specialty'
data1=[a1,a2,a3,a4]
expl = [0,0.1,0,0]
plt.pie(data1,labels=lab1,
        autopct='%1.2f%%',
        shadow=True,
        explode=expl)
plt.savefig('d:/picture/学科.png')
plt.show()
#老师
b1=sum(df['视频标签'].str.contains('张宇'))
b2=sum(df['视频标签'].str.contains('汤家凤'))
b3=sum(df['视频标签'].str.contains('何凯文'))
b4=sum(df['视频标签'].str.contains('朱伟'))
b5=sum(df['视频标签'].str.contains('张雪峰'))
b6=sum(df['视频标签'].str.contains('徐涛'))
print(b1,b2,b3,b4)
lab1='zy','tjf','hkw','zw','zxf','xt'
data1=[b1,b2,b3,b4,b5,b6]
expl = [0,0.1,0,0,0,0]
plt.pie(data1,labels=lab1,
        autopct='%1.2f%%',
        shadow=True,
        explode=expl)
plt.savefig('d:/picture/老师.png')
plt.show()
#专业
c1=sum(df['视频标签'].str.contains('计算机'))
c2=sum(df['视频标签'].str.contains('医学'))
c3=sum(df['视频标签'].str.contains('经济'))
c4=sum(df['视频标签'].str.contains('心理学'))
print(c1,c2,c3,c4)
lab1='IT','M','E','P'
data1=[c1,c2,c3,c4]
expl = [0,0.1,0,0]
plt.pie(data1,labels=lab1,
        autopct='%1.2f%%',
        shadow=True,
        explode=expl)
plt.savefig('d:/picture/专业.png')
plt.show()
#3.2绘制条形图
count=0
for i in df['视频时长']:
    x=float(i[0:2])*float(i[3:5])
    df.iloc[count,0]=str(x)
    count=count+1
print(count)
#将时间序列改成datetime64[ns]属性以便时间截取
df.index=pd.to_datetime(df.index)
def _hot_index_fun(x):
    s=[]
    s.append(sum(df['2017-03-01':'2017-6-30']['hot_index']>x))
    s.append(sum(df['2017-07-01':'2017-8-31']['hot_index']>x))
    s.append(sum(df['2017-09-01':'2017-12-24']['hot_index']>x))
    return s

t=_hot_index_fun(0.03)

df['视频时长']=df['视频时长'].astype('float64')

tl=[]
tl.append(sum(df['2017-03-01':'2017-6-30']['视频时长']))
tl.append(sum(df['2017-07-01':'2017-8-31']['视频时长']))
tl.append(sum(df['2017-09-01':'2017-12-24']['视频时长']))
def sub1(name):
    count=0
    tl1=[]
    sum0=0
    for i in df['2017-03-01':'2017-6-30']['视频标签'].str.contains(name):
        if i==True:
            print(df.iloc[count,0])
            sum0=sum0+df.iloc[count,0]
        count=count+1
    tl1.append(sum0)
    sum0=0
    for i in df['2017-07-01':'2017-8-31']['视频标签'].str.contains(name):
        if i==True:
            print(df.iloc[count,0])
            sum0=sum0+df.iloc[count,0]
        count=count+1
    tl1.append(sum0)
    sum0=0
    for i in df['2017-09-01':'2017-12-24']['视频标签'].str.contains(name):
        if i==True:
            print(df.iloc[count,0])
            sum0=sum0+df.iloc[count,0]
        count=count+1
    tl1.append(sum0)
    print(count)
    return tl1

tl1=sub1('数学')
tl2=sub1('英语')
tl3=sub1('政治')
def countlabs(name):
    sbj1=[]
    sbj1.append(sum(df['2017-03-01':'2017-6-30']['视频标签'].str.contains(name)))
    sbj1.append(sum(df['2017-07-01':'2017-8-31']['视频标签'].str.contains(name)))
    sbj1.append(sum(df['2017-09-01':'2017-12-24']['视频标签'].str.contains(name)))
    return sbj1
    
sbj1=countlabs('数学')
sbj2=countlabs('英语')
sbj3=countlabs('政治')

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
font = {'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 16  
        }  
index = np.arange(3)
bar_width = 0.3
plt.bar(index, sbj1, width=0.3 , color='g',label=u'数学')
plt.bar(index+bar_width, sbj2, width=0.3 , color='r',label=u'英语')
plt.bar(index+bar_width+bar_width, sbj3, width=0.3 , color='b',label=u'政治')
plt.xlabel(u'阶段（3-6，7-8，9-12）',fontdict=font)
plt.ylabel(u'视频个数',fontdict=font)
plt.title(u'三个阶段各个科目视频个数变化',fontdict=font)
plt.legend(loc=2) 
plt.savefig('d:/picture/三个阶段各个科目变化.png')
plt.show()

index = np.arange(3)
bar_width = 0.3
plt.bar(index, tl1, width=0.3 , color='g',label=u'数学')
plt.bar(index+bar_width, tl2, width=0.3 , color='r',bottom=sbj1,label=u'英语')
plt.bar(index+bar_width+bar_width, tl3, width=0.3 , color='b',bottom=sbj2,label=u'政治')
plt.xlabel(u'阶段（3-6，7-8，9-12）',fontdict=font)
plt.ylabel(u'视频时长',fontdict=font)
plt.title(u'三个阶段各个科目变化视频时长变化',fontdict=font)
plt.legend(loc=2) 
plt.savefig('d:/picture/时长比.png')
plt.show()

index = np.arange(3)
bar_width = 0.3
plt.bar(index, t, width=0.3 , color='g',label=u'热度指标变化')
plt.xlabel(u'阶段（3-6，7-8，9-12）',fontdict=font)
plt.ylabel(u'hot_index',fontdict=font)
plt.title(u'热度指标变化',fontdict=font)
plt.legend(loc=2) 
plt.savefig('d:/picture/hot.png')
plt.show()
#3.3绘图词云
print(df['视频标签'])
labs=str(list(df['视频标签']))
labs=''.join(jieba.cut(labs))
cloud_mask = np.array(Image.open("d:/timg.jpg"))
wordcloud =wc(font_path='c:/Users/Windows/fonts/simkai.ttf',
              stopwords=['考试','数学','搞笑'],
              max_words=50,
              background_color="white",
              mask=cloud_mask
              )
wcd=wordcloud.generate(labs)
image_colors=ImageColorGenerator(cloud_mask)
wordcloud.recolor(color_func=image_colors)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
wcd.to_file("d:/picture/kaoyan.png")
#散点图
plt.plot_date(df.index,df['hot_index'])
plt.savefig('d:/picture/data_hot_index.png')
plt.show()

