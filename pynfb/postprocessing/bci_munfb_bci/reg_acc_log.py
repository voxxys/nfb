import pandas as pd
import seaborn as sns
import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
from scipy import stats

df = pd.read_excel(r'C:\Users\Nikolai\Desktop\bci_nfb_bci\bci_nfb_bci\acc-points.xlsx')
df['score'] = df['score']*100

print(df)

def func(x, a, b, c, d):
    return c*np.tanh(a*x+b) + d

xdata = df['acc'].as_matrix()
ydata = df['score'].as_matrix()
ydata = ydata[np.argsort(xdata)]
xdata = xdata[np.argsort(xdata)]
xm = xdata.mean()
xs = xdata.std()
ym = ydata.mean()
ys = ydata.std()
#plt.plot(xdata, ydata, 'o', label='data')

#xdata = (xdata - xm) / xs
#ydata = (ydata - ym) / ys




#popt, pcov = curve_fit(func, xdata, ydata)
popt, pcov = curve_fit(func, xdata, ydata, p0=[44.3312025231, -35.9289164708, 11.6096937442, 24.0851942223])
a, b, c, d = popt
print(popt)
ae, be, ce, de = np.sqrt(np.diag(pcov))
print(a/xs, b-a*xm/xs, c*ys, ym+ys*d)
#plt.plot(xdata*xs+xm, func(xdata, *popt)*ys+ym, 'r-',

sns.regplot('acc', 'score', df, order=1, label='$scores = k \cdot acc + const$\np-value($k$) = {:.2e}'.format(stats.linregress(df['acc'], df['score']).pvalue))
plt.plot(xdata, func(xdata, *popt),
         label='$scores = c \cdot tanh(a\cdot acc +b)+d$\na={:.1f}$\pm${:.1f}, b={:.1f}$\pm${:.1f}, c={:.1f}$\pm${:.1f}, d={:.1f}$\pm${:.1f}'.format(a, ae, b, be, c, ce, d, de))




plt.xlabel('BCI accuracy')
plt.ylabel('Game scores')
plt.legend()
plt.show()