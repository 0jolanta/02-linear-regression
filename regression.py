#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame()
df['X'] = [1, 2, 3, 4, 5]
df['Y'] = [4, 6, 9, 11, 18]
df


# In[2]:


plt.scatter(df['X'], df['Y'], label='Wartości Niezależne')
plt.xlabel('Wartości X')
plt.ylabel('Wartości Y')
plt.legend()
plt.show()


# In[3]:


#Zadanie 2 - średnia arytmetyczna
def srednia(zbior):
    return float(zbior.sum()) / len(zbior)

Mx = srednia(df['X'])
My = srednia(df['Y'])

print("Mx: ", Mx)
print("My: ", My)


# In[6]:


# Zadanie 3 - odychylenie standardowe z próby
from math import sqrt

def odchylenie(zbior, srednia):
    licznik = 0
    for elem in zbior:
        licznik += (elem - srednia) * (elem - srednia)
    return sqrt(licznik / (len(zbior) - 1))

Sx = odchylenie(df['X'], Mx)
Sy = odchylenie(df['Y'], My)

print("Sx: ", Sx)
print("Sy: ", Sy)


# In[8]:


# Zadanie 4 - Współczynnik korelacji
n = len(df['X'])
vr = pd.DataFrame(df[:])
vr['y2'] = df['Y'] * df['Y']
vr['xy'] = df['X'] * df['Y']
vr['x2'] = df['X'] * df['X']
vr['y2'] = df['Y'] * df['Y']
vr.loc['Σ'] = vr.sum()

print("n = ", n)
print()
print(vr)


# In[13]:


def wsp_korelacji_pearsona(n, ΣX, ΣY, Σxy, Σx2, Σy2):
    return ( (n * Σxy - ΣX * ΣY) /
             (sqrt((n * Σx2 - ΣX**2) * (n * Σy2 - ΣY**2))) )

r = wsp_korelacji_pearsona(n, vr['X']['Σ'], vr['Y']['Σ'], vr['xy']['Σ'], vr['x2']['Σ'], vr['y2']['Σ'])
print("r = ", r)


# In[27]:


import numpy as np

a = r * (Sy / Sx)
b = My - (b * Mx)


print("a = ", a)
print("b = ", b)

def najlepiej_pasujaca_linia(x):
    return (a * x) + b

x = np.linspace(0, 5, 1000)
plt.scatter(df['X'], df['Y'], label='Wartości Niezależne')
plt.plot(x, najlepiej_pasujaca_linia(x), 'r', label='Najlepiej Pasująca linia')
plt.xlabel('Wartości X')
plt.ylabel('Wartości Y')
plt.legend()
plt.show()


# In[38]:


df = df.append({'X': 6, 'Y': np.nan}, ignore_index=True)
df


# In[41]:


def wsp_determinacji_r_kwadrat(a,b, df, My):
    SSm = 0.00
    SSt = 0.00
    for _, rzad in df.iterrows():
        x = rzad['X']
        y = rzad['Y']
        y_p = przewidz_y(x, a, b)
        SSm += (y_p - My) **2
        SSt += (y - My) ** 2
    return SSm / SSt

R2 = wsp_determinacji_r_kwadrat(a, b, df, My)
print("R2 = ", R2) 


# In[42]:


df = df.append({'X': 20, 'Y': np.nan}, ignore_index=True)
df.count()['Y']

def RegresjaLiniowa(inp):
    y_dlug = inp.count()['Y']
    out = inp[:]
    inp = inp.dropna()
    
    Mx = srednia(inp['X'])
    My = srednia(inp['Y'])
    
    Sx = odchylenie(inp['X'], Mx)
    Sy = odchylenie(inp['Y'], My)

    n = len(df['X'])
    vr = pd.DataFrame(inp[:])
    vr['y2'] = df['Y'] * df['Y']
    vr['xy'] = df['X'] * df['Y']
    vr['x2'] = df['X'] * df['X']
    vr['y2'] = df['Y'] * df['Y']
    vr.loc['Σ'] = vr.sum()
    r = wsp_korelacji_pearsona(n, vr['X']['Σ'], vr['Y']['Σ'], vr['xy']['Σ'], vr['x2']['Σ'], vr['y2']['Σ'])
    b = r * (Sy / Sx)
    a = My - (b * Mx)
   

    dokl = wsp_determinacji_r_kwadrat(b, a, inp, My)
    
    for i, rzad in out.iterrows():
        if np.isnan(rzad['Y']):
            
            df.at[i, 'Y'] = przewidz_y(rzad['X'], b, a)
    
    return out, dokl

wynik, dokladnosc = RegresjaLiniowa(df)

print("Dokładność = ", dokladnosc)
print(wynik)


# In[ ]:




