# -*- coding: utf-8 -*-

import pandas as pd

# importing and treating data
base = pd.read_csv('ALP.csv', index_col=0, parse_dates=(True))
data = base.loc[ :'2021-06-29']
alpine = data['ALPINE FIM']
cdi = data['CDI']

##### 2D LINE PLOT GRAPH #####
import matplotlib.pyplot as plt

### general settings
font_size = 15

### Figure
## instancing the figure
fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)

## Style
# plt.style.use('fivethirtyeight') # style

## Figure title
pass

# ## Source 
# fig.text(x = 0, y = -0.1,         # coordinates in figure
#          transform = ax1.transAxes, # changes from plane coord to figcoord
#          s ='Fonte : SOLE capital')

### ploting the lines ###
## 2D Line 1
plt.plot(cdi,                  # data input
         color = '#D59D4A', 
         # alpha=0.5,          # set the 'strenght' of the color
         linestyle = '-',
         linewidth = 2,
         label = 'CDI',
         ) 
## 2D Line 2
plt.plot(alpine,               # data input
         color = '#307D9F', 
         # alpha=0.5,          # set the 'strenght' of the color
         linestyle = '-',
         linewidth = 2,
         label = 'Alpine FIM',
         ) 

### setting axes parameters (for single axes = ax1) ###
# https://matplotlib.org/stable/api/axes_api.html

## Axes background color
ax1.set_facecolor('#E5E5E7') # color in hex

## Axes Titles
ax1.set_title('Historical Performance',
              loc = 'center', # high-level positioning arg
              fontdict = {'fontsize': font_size+10,
                          'color': '#307D9F',
                          'fontweight': 'bold',
                          },
              pad = 12 # distancing from the graph
              )

## Grid 
ax1.yaxis.grid(which="major", color='w', linewidth=1)
ax1.xaxis.grid(which="major", color='w', linewidth=1)

## Spines
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

## Axis size & orientation
ax1.set_xlim(left=None, right=None) 
ax1.set_ylim(bottom=None, top=None)
# ax1.invert_xaxis()     
# ax1.invert_yaxis()
# ax1.autoscale(enable=True, axis='both', tight=None)

## Axis Labels
ax1.set_ylabel('Return (%)', # the label
               loc = 'center',
               size= font_size
               )

### Ticks ###
## Tick Marks
ax1.tick_params(axis = 'both', # both x and y axis affected
               which = 'major', # to wich tick mark apply
               color = 'grey',
               labelsize = font_size, # font size 
               )

## Tick marks - Custom axis setting
ax1.tick_params(axis = 'x',        
                # rotation = 20,
                ) # only x ticks

## Legend
ax1.legend(loc = 'best', # location (tuple can be used (x,y))
           fontsize = font_size,
           labelcolor = 'k',      # color of legend text
           facecolor = 'inherit', # color of the background
           frameon = True,        # put a frame behind legend
           framealpha = 1 ,       # transparency
           )

### Text & Annotation & Misc ###
pass

# ## Arrow
# ax1.annotate('text to show',           # text to show 
#              (10, 16),               # coordinates xy
#              textcoords = (8, 15),  # coordinates of textbox
#              # Arrow between xy and xytextcoords 
#              arrowprops= {'arrowstyle': '->'} # see doc for options        
#              )


