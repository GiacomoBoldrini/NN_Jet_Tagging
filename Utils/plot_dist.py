import matplotlib.pyplot as plt 
import numpy as np
import ROOT 

class plotting_distribution():
    
    def plot(features, gluons_data, quarks_data, Save = False, outpath = "./"):
        """
            Takes as input a list of features names, gluons data and quarks data as numpy 
            array. If Save option is True then plots will be saved under the outpath input
            otherwise "./" is the default.
        """
        
        if (len(features) != gluons_data.shape[1]) or (gluons_data.shape[1] != quarks_data.shape[1]):
            print("Check features and data length.")
            return 
        
        for i in range(len(features)):
            gluon = gluons_data[:,i]
            quarks = quarks_data[:,i]
            plt.figure(figsize=(13,8))
            # then plot the right quantity for the reduced array
            plt.hist(gluon , 50, density=True, histtype='step', fill=False, linewidth=1.5, label = "gluon")
            plt.hist(quarks , 50, density=True, histtype='step', fill=False, linewidth=1.5, label = "quarks")
            plt.yscale('log', nonposy='clip')    
            plt.legend(fontsize=12, frameon=False)  
            plt.xlabel(features[i], fontsize=15)
            plt.ylabel('Prob. Density (a.u.)', fontsize=15)
            if Save:
                plt.savefig(outpath+"{}.pdf".format(features[i]))