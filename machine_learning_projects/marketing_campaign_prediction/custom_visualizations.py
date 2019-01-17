## Created by Paul de Fusco
## Custom Visualization Methods

#%load_ext autoreload

#%autoreload 2

import matplotlib.pyplot as plt
import seaborn as sns

def show_countplots(df, features, hue):

    rows = 0
    if len(features)%2:
        rows += 1
    
    rows=int((len(features)+rows)/2)
    
    cols = 2
    
    fig, axes = plt.subplots(rows,cols, sharex=False, sharey=False, figsize=(30,10*rows), squeeze=False)

    for ax, feature in zip(axes.flat, features):

        sns.countplot(x=feature, y=None,
                    hue=hue,
                    data=df,
                    #ci="sd",
                    ax=ax)
        ax.set_title("Histogram for %s" %(str(feature)))
        ax.legend

    plt.show()