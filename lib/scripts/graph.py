
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy import signal
from utils import read_edf_by_path
import matplotlib
from pandas import read_excel,read_csv
import matplotlib.pyplot as plt

raw = read_edf_by_path(f'data/0_raw/20-Aug-A1-0-PF.edf')
X = raw.get_data(picks='EEG')

# df = read_csv(f'final.csv')
# df[df == 'W'] = 2
# df[df == 'P'] = 1
# df[df == 'S'] = 0
# df[df == 'X'] = -1
# df = df['0']
wave_duration = 8000
sample_rate = 500
samples = wave_duration*sample_rate

x = np.linspace(0, wave_duration, samples, endpoint=False)
y = X[0][:samples]

## plotting
x_center = 45
y_spread = .00025
fig, axes = plt.subplots(nrows=2,ncols=1,gridspec_kw={'height_ratios': [1, 8]})
cvals  = [-1,0,1, 2]
colors = ["black","purple","red","blue"]
norm=plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
plt.xlim([x_center-45,x_center+45])
plt.ylim([0-y_spread,0+y_spread])
# label_locs = np.arange(.5,len(df),1)*10
# axes[0].scatter(label_locs,df,c=df,cmap=cmap,norm=norm,s=200)
l = axes[1].plot(x, y, linewidth=.3)
# axes[0].axis('off')
# q = 20
# samples_decimated = int(samples/q)
# ydem = signal.decimate(y, q)
# xnew = np.linspace(0, wave_duration, samples_decimated, endpoint=False)
# axes[0].scatter(label_locs,df,c=df,cmap=cmap,norm=norm,s=20)
# axes[0].scatter(label_locs,df)

# m = axes[1].plot(xnew,ydem,linewidth=.5 )
# del df
del raw
def update():
    axes[0].axis([x_center-45,x_center+45,-1.5,4])
    axes[1].axis([x_center-45,x_center+45,0-y_spread,0+y_spread])
    fig.canvas.draw_idle()

def on_key_release(event):
    global x_center
    global y_spread
    global labels
    if(event.key == 'left'):
        x_center = x_center-10
        print()
        update()
    elif(event.key == 'right'):
        x_center = x_center+10
        update()
    elif(event.key == 'down'):
        y_spread = y_spread - .00001
        update()
    elif(event.key == 'up'):
        y_spread = y_spread + .00001
        update()

fig.canvas.mpl_connect('key_press_event', on_key_release)

plt.show()


