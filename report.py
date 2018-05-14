import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset import _BRANDS
class ReportGenerator:
    def __init__(self, dirname, target, input, scaler_y = None, scaler_x = None):
        self.dirname = dirname
        self.scaler_y = scaler_y
        self.scaler_x = scaler_x
        self.target = target
        self.input = input

    def assess(self,output,target):
        res = []
        size = max(output.size, target.size)
        sum = 0
        for i in range(size):
            diff = abs(output[i][0] - target[i][0])/max(output[i][0], target[i][0])
            sum = sum + diff
            res.append([output[i][0], target[i][0], diff])
        res[0].append(sum/size)
        for i in range(1,size):
            res[i].append(0)
        return (res,sum/size)

    def generate(self, lmodels):
        models_errors = []
        idx = 0
        for descr,model in lmodels.items():
            # output = model.predict(self.input if self.scaler_x is None else self.scaler_x.transform(self.input), batch_size=1, verbose=0)
            # target = self.target
            output = self.scaler_y.inverse_transform(model.predict(self.input if self.scaler_x is None else self.scaler_x.transform(self.input), batch_size=1, verbose=0))
            target = self.scaler_y.inverse_transform(self.target)
            # plt.title(descr)
            # plt.plot(range(len(output)), output, 'ro', range(len(output)), target, 'bo')
            # plt.savefig(self.dirname+'/plot_'+str(idx)+'.png')

            res,error = self.assess(output,target)
            df = pd.DataFrame(res)
            models_errors.append([error,'/model'+str(idx)])
            df.to_csv(self.dirname+'/model'+str(idx)+'.csv')
            model.save(self.dirname+'/model'+str(idx)+'.mdl')
            idx = idx + 1
        total_df = pd.DataFrame(models_errors)
        total_df.to_csv(self.dirname+'/total.csv')

    def ensemble_generate(self,lmodels):
        # brands = [el.split('_') for el in _BRANDS]
        for idx,model in enumerate(lmodels):
            x = self.input if self.scaler_x is None else self.scaler_x.transform(self.input)
            res = model.predict(x)
            outputs = np.apply_along_axis(self.scaler_y.inverse_transform,axis=0,arr=res)
            target = self.target.astype('float64')
            # plt.title(brands[idx])
            # plt.plot(range(len(outputs)), outputs, 'ro', range(len(outputs)), target, 'bo')
            # #plt.savefig(self.dirname+'/plot_'+brands[idx]+'.png')
            # plt.show()
            df = pd.DataFrame(np.c_[outputs.reshape(outputs.shape[1],outputs.shape[0]),target])
            df.to_csv(self.dirname+'/model'+str(idx)+'.csv')
