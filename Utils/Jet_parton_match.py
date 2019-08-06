import math as mt
import numpy as np

class Jet_parton_match:
    
    def delta_phi(a1,a2):
        #computing difference between two angles
        #taking in account ciclycity
        return np.sign(a1-a2)*(180 - abs(abs(a1 - a2) - 180))
    
    def Delta_R(parton, jet):
        """
            Computing Delta R: square root of sum of squares of delta_phi
            and delta_eta between parton and jet.
            
        """
        return mt.sqrt((jet.Eta-parton.Eta)**2+(Jet_parton_match.delta_phi(jet.Phi,parton.Phi))**2)


    def high_pt_jet_match(partons, jets):
        """
            Computing match between the two most energetic
            jets in the event. This is an approximation.
        """
        #selecting two highest PT jets
        jets = sorted(jets, key= lambda x: x.PT)
        jets = jets[0:2]
        
        p_idx = list(np.arange(0, len(partons), 1))
        j_idx = list(np.arange(0, len(jets), 1))
        parton_jet = []
        
        #checking for DeltaR compatibility
        for i in range(len(p_idx)):
            parton_jet_R = []
            for p in p_idx:
                parton = partons[p]
                for j in j_idx:
                    jet = jets[j]
                    parton_jet_R.append([p, j, Jet_parton_match.Delta_R(parton,jet)])
        
            parton_jet_R = sorted(parton_jet_R, key= lambda x:x[2])
            
            if (parton_jet_R[0][2] < .7 ):
                parton_jet.append([partons[parton_jet_R[0][0]], jets[parton_jet_R[0][1]]])
                if len(p_idx) != 1:
                    p_idx.pop(parton_jet_R[0][0])
                    j_idx.pop(parton_jet_R[0][1])

        if len(parton_jet) == len(partons):
            return parton_jet
                
        else:
            return False



