#!/usr/bin/env python
# coding: utf-8

# 数据从kaggle中获得

# 数据挖掘就是找出数据中的规律或趋势，然后再将结果应用到实际情况中，从而创造价值。那么这两份数据我觉得可以从中找出那些成功app的特性，并基于此设计出其他的热门app

# 程序在开发之前都是要先确定方向的，而方向是非常重要的，这可以影响到app在开发并投入市场后的受欢迎程度，那么我就可以基于这份数据来找出方向

# In[96]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import warnings
warnings.filterwarnings("ignore")


# In[97]:


googleapp = pd.read_csv("C:/Users/zhang/Desktop/google-play-store-apps/googleplaystore.csv")


# In[98]:


googleapp.head()


# 数据中有一个属性Category，这个是用来标注app类型的，我可以通过统计这个属性中app的类型来查看哪些app更受欢迎，因为从整体市场中哪些app所占市场份额越大，那么就证明哪些app会更受欢迎一些

# In[99]:


googleapp['Category'].unique()


# 在这些类型中有一个1.9，可以认为这是一个错误的值，需要进行处理

# In[100]:


googleapp[googleapp['Category'] == '1.9']


# In[101]:


googleapp.loc[10472]


# 上面这样的情况所对应的数据只有一条，所以最简单的方式就是直接删除，但是通过观察，我发现这条数据中每个属性的内容都是错乱的，如果重新调整一下就可以了

# In[102]:


googleapp.loc[10472] = googleapp.loc[10472].shift()
googleapp['App'].loc[10472] = googleapp['Category'].loc[10472]
googleapp['Category'].loc[10472] = np.nan
googleapp.loc[10472]


# 现在这条数据正常多了

# 数据中含有很多重复数据，需要去掉

# In[103]:


googleapp[googleapp.duplicated()].count()


# In[104]:


googleapp = googleapp.drop_duplicates()
googleapp[googleapp.duplicated()].count()


# In[105]:


googleapp.info()


# 数据中的属性都是离散类型的，从所有属性可以大概的判断出Rating、Reviews、Size、Installs、Price这些属性对于是否受欢迎是有影响力的，而其他的属性与受欢迎程度无关

# 下面要转换Rating、Reviews、Size、Installs、Price的数据类型才能进行后续的其他分析操作

# In[106]:


googleapp['Rating'].unique()


# In[107]:


googleapp['Rating'] = pd.to_numeric(googleapp['Rating'],errors='coerce')


# In[108]:


googleapp['Reviews'].unique()


# In[109]:


googleapp['Reviews'] = pd.to_numeric(googleapp['Reviews'],errors='coerce')


# In[110]:


googleapp['Size'].unique()


# 从上面的显示结果可以发现，其中有一个"Varies with device"，这个值可以用缺失值来替代，否则会影响数据分析

# In[111]:


googleapp['Size'].replace('Varies with device',np.nan,inplace=True)


# In[113]:


#数据转换
googleapp['Size'] = googleapp['Size'].str.extract(r'([\d\.]+)',expand=False).astype(float) * googleapp['Size'].str.extract(r'([kM]+)',expand=False).fillna(1).replace(['k','M'],[1,1000]).astype(int)


# In[115]:


googleapp['Installs'].unique()


# In[116]:


googleapp['Installs'] = googleapp['Installs'].str.replace(r'\D','').astype(float)


# In[118]:


googleapp['Price'].unique()


# In[119]:


googleapp['Price'] = googleapp['Price'].str.replace('$','').astype(float)


# 数据处理结束后可以对数据进行可视化了

# 为了找出哪些类型的app占领的市场份额比较大，可以对Category中的不同类型的数量进行统计，并可视化出来

# In[123]:


plt.figure(figsize=(10,10))
g = sns.countplot(y='Category',data=googleapp,palette="Set2")
plt.title("Number of each Category",size=20)


# 从上面的可视化结果可以看出Family、Game和Tools占的是最多的

# 现在可以知道哪些类型的app占领的市场份额更大，但是Installs同样影响着受欢迎程度，所以还需要观察每个Category所对应的Installs的情况

# In[126]:


plt.figure(figsize=(10,10))
g = sns.barplot(x='Installs',y='Category',data=googleapp,capsize=.6)
plt.title("Installations in each Category",size=20)


# 根据可视化结果可以发现社交类的app的下载量是最大的

# 下面筛选一下满足一定条件的Installs的Category都有哪些

# In[128]:


googleapp[googleapp[['Installs']].mean(axis=1) > 1e5]['Category'].unique()


# 上面是通过Installs筛选出来的大于10万的类型的app

# In[130]:


plt.figure(figsize=(10,10))
plt.scatter(x='Rating',y='Installs',data=googleapp,color='blue')
g = sns.lineplot(x='Rating',y='Installs',data=googleapp,color='red')
plt.yscale('log')
plt.xlabel('Rating')
plt.ylabel('Installs')
plt.title('Rating--Installs',size=20)


# 上面将散点图和线段图都整合到了一起，其中散点图可以看出数据的整体的分布情况，而线段图是数据的整体变化趋势。

# 从散点图可以看出app的整体评分都集中在3.0以上；而通过线段图可以看出Rating与Installs之间整体呈正向关系，但是在Rating=5的app下载量却偏低，其实是可以这么理解的

# 一般来说评级越高证明app就越受欢迎，那么下载量也就相对更多，但是也不尽然，因为有些时候，哪怕评级很高也不一定就有很高的下载量，这是因为哪怕下载量很少，但如果都是积极的评论，那么评分依然可以很高，但不一定下载量就会很高

# 下面是Reviews

# In[131]:


g = sns.lmplot(y='Installs',x='Reviews',data=googleapp,size=(10))
plt.xscale('log')
plt.yscale('log')
plt.title("Reviews--Installs",size=20)


# 从这个可视化结果可以看出，Installs与Reviews之间呈正相关

# 观察Size属性

# In[133]:


plt.figure(figsize=(10,10))
g = sns.boxplot(x='Installs',y='Size',data=googleapp)
g.set_xticklabels(g.get_xticklabels(),rotation=40,ha='right')
plt.title("Installs--Size",size=20)


# In[137]:


plt.figure(figsize=(10,10))
g = sns.lineplot(x='Reviews',y='Installs',data=googleapp,color='red')
plt.xscale('log')
plt.yscale('log')


# 从这个可视化可以看出随着Installs的增加，Size也会随之增加，这是因为只有那些足够美观的app，功能全面的app才更受欢迎，而如果要想设计出足够美观和功能全面的app的话，那么软件的Size就一定大，换句话说，如果软件要想受欢迎，那么一定要足够美观和功能全面

# 观察Content Rating

# In[138]:


plt.figure(figsize=(10,10))
g = sns.barplot(y="Installs",x='Content Rating',data=googleapp,capsize=.5)
g.set_xticklabels(g.get_xticklabels(),ha='left')
plt.title('Content Rating--Installs',size=20)


# In[142]:


labels = googleapp['Content Rating'].unique()
explode = (0.1,0,0,0,0)
size = []
for label in labels:
    size.append(googleapp[googleapp['Content Rating'] == label]['Installs'].mean())

#merge Unrated and Adult
labels[4] = 'Unrated &\n Adults only 18+'
labels = np.delete(labels,5)
size[4] = size[4] + size[5]
size.pop()

#pie
plt.figure(figsize=(10,10))
colors = ['#ff6666','#66b3ff','#99ff99','#ffcc99','#df80ff']
plt.pie(size,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90)
plt.axis('equal')
plt.title("Percentage of Installs",size=20)


# 从这两个图都可以看出来，只有常规的app下载量是最大的，那些成人级别的下载量很少

# In[144]:


plt.figure(figsize=(10,10))

labels = ['Download less than 100000','Download more than 100000']
size = []
size.append(googleapp['App'][googleapp['Installs'] < 1e5].count())
size.append(googleapp['App'][googleapp['Installs'] >= 1e5].count())

labels_inner = ['Free','Paid','Free','Paid']
size_inner = []
size_inner.append(googleapp['Type'][googleapp['Type'] == 'Free'][googleapp['Installs'] < 1e5].count())
size_inner.append(googleapp['Type'][googleapp['Type'] == 'Paid'][googleapp['Installs'] < 1e5].count())
size_inner.append(googleapp['Type'][googleapp['Type'] == 'Free'][googleapp['Installs'] >= 1e5].count())
size_inner.append(googleapp['Type'][googleapp['Type'] == 'Paid'][googleapp['Installs'] >= 1e5].count())

colors = ['#99ff99', '#66b3ff']
colors_inner = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']

explode = (0,0) 
explode_inner = (0.1,0.1,0.1,0.1)

#outer pie
plt.pie(size,explode=explode,labels=labels, radius=3, colors=colors)
#inner pie
plt.pie(size_inner,explode=explode_inner,labels=labels_inner, radius=2, colors=colors_inner)
       
#Draw circle
centre_circle = plt.Circle((0,0),1.5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.tight_layout()


# 从结果可以看出下载量少于10万和下载量大于10万的数量大致是相同的；最后还有Price属性，这个属性直接选择Free就可以了

# In[145]:


plt.figure(figsize=(10,10))
g = sns.lineplot(x='Price',y='Installs',data=googleapp)


# 上面通过可视化可以看出price整体都是0，也就是Free，下面将price控制在0-10之间再继续观察

# In[148]:


plt.figure(figsize=(10,10))
g = sns.lineplot(x='Price',y='Installs',data=googleapp)
g.set(xlim=(0,10))


# app大多都是免费的，只有少数是收费的，且价格不贵；下载量基本遵循价格的增加而下降，所以这里就明白了，如果打算收费，则应该将app定价到1以下为最好

# In[156]:


from sklearn.feature_extraction.text import CountVectorizer

corpus=list(googleapp['App'])
vectorizer = CountVectorizer(max_features=50, stop_words='english')
X = vectorizer.fit_transform(corpus)
names=vectorizer.get_feature_names()
values=X.toarray().mean(axis=0)

plt.figure(figsize=(20,20))
sns.barplot(x=values, y=names, palette="viridis")
plt.title('Top 50',size = 30)


# In[201]:


words = []
for corpu in corpus:
    for name in names:
        if name in corpu:
            words.append(name)

cloud_words = str()
cloud_words += ' '.join(word for word in words)


# In[202]:


from wordcloud import WordCloud
wordcloud = WordCloud(width=800,height=500,random_state=21,                      max_font_size=110).generate(cloud_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')


# 上面通过条形图和词云的形式展示了出现频率最高的单词

# 最后的分析结果就是，可以开发一个社交类app，或是其他满足条件的都可以，但是社交类app的下载量是最大的；其次app本身必须要足够美观，且功能完备；设定使用限制必须要在常规级别，不可以是那种成人级别的；最后是价格，最好设定为免费，如果要收费的话，也不要高于1美元；最后是app的名称，而名称在一定程度上可以反映app本身的功能和定位，所以app可以开发成一些包含video、game、photo、book等的app

# 如果开发的app满足以上条件，那么有更大可能性可以提升下载量

# In[ ]:




