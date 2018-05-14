import pandas as pd
import matplotlib.pyplot as plt
filedir = './output/'

_BRANDS = [
    'BMW',
    'Dodge',
    'Ferrari',
    'Hyundai',
    'Kia',
    'Mercedes-Benz',
    'Mini',
    'Peugeot',
    'Toyota',
    'Volkswagen'
]
for brand in _BRANDS:
    df = pd.read_csv(filedir+brand+'/model0.csv')
    print(df)
    plt.title(brand)
    plt.plot(df.index, df['0'], 'ro', df.index, df['1'], 'bo')
    plt.savefig(filedir+brand+'/model0.png')
    plt.clf()
    plt.cla()
    plt.close()