import math as mt
import numpy as np
import pandas as pd
import ROOT
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from Utils.Jet_parton_match import Jet_parton_match

class image_creator():

    def Jet_Pt_dist(data_path, parton_to_match = 'q', three_d = False):
        
        chain = ROOT.TChain("Delphes")
        for file in data_path:
            chain.Add(file)

        treeReader = ROOT.ExRootTreeReader(chain)
        treeReader = ROOT.ExRootTreeReader(chain)
        numberOfEntries = treeReader.GetEntries()
        print(numberOfEntries)

        branchJet = treeReader.UseBranch("Jet")
        branchFatJet = treeReader.UseBranch("FatJet")
        branchMuon = treeReader.UseBranch("Muon")
        branchElectron = treeReader.UseBranch("Electron")
        branchMET = treeReader.UseBranch("MissingET")
        branchPhoton = treeReader.UseBranch("Photon")
        branchParticle = treeReader.UseBranch("Particle")
        branchEFlowTrack = treeReader.UseBranch("EFlowTrack")
        branchEFlowTower = treeReader.UseBranch("EFlowTower")
        branchEFlowMuon = treeReader.UseBranch("EFlowMuon")
        branchEFlowPhoton = treeReader.UseBranch("EFlowPhoton")
        branchEFlowNeutralHadron = treeReader.UseBranch("EFlowNeutralHadron")
        branchTower = treeReader.UseBranch("Tower")

        images_partons = []

        for entry in tqdm(range(0, numberOfEntries)):
            
            
            treeReader.ReadEntry(entry)
            
            #building partons
            partons = []
            for p in branchParticle:
                
                #light quark selection
                if parton_to_match == 'q':
                    if (p.Status == 23) and (abs(p.PID) <= 8):
                        partons.append(p)
            
                #gluons jet selection
                if parton_to_match == 'g':
                    if (p.Status == 23) and (abs(p.PID) == 21):
                        partons.append(p)
            
            if branchJet.GetEntries() >= 2:
                
                jets = [i for i in branchJet]
                
                #checking if match max DeltaR with highest Pt jets
                association = Jet_parton_match.high_pt_jet_match(partons, jets)
                
                #if no complete match between highest PT and partons
                #move on
                if not association:
                    continue
            
                associated_jets = [row[1] for row in association]
                
                
                #building jets images
                if not three_d:
                    
                    for j in range(branchJet.GetEntries()):
                        
                        jet_im = []
                        particles = branchJet.At(j).Constituents
                        jet_Eta = branchJet.At(j).Eta
                        jet_Phi = branchJet.At(j).Phi
                        
                        for p in range(particles.GetEntries()):
                            
                            if(type(particles.At(p)) == ROOT.GenParticle ):
                                four_mom_p = particles.At(p).P4()
                                pt = four_mom_p.Pt()
                                jet_im.append([jet_Eta-particles.At(p).Eta, jet_Phi-particles.At(p).Phi, pt ])
                            
                            if(type(particles.At(p)) == ROOT.Track ):
                                four_mom_p = particles.At(p).P4()
                                pt = four_mom_p.Pt()
                                jet_im.append([jet_Eta-particles.At(p).Eta, jet_Phi-particles.At(p).Phi, pt ])
                            
                            if(type(particles.At(p)) == ROOT.Tower ):
                                four_mom_p = particles.At(p).P4()
                                pt = four_mom_p.Pt()
                                jet_im.append([jet_Eta-particles.At(p).Eta, jet_Phi-particles.At(p).Phi, pt ])
                    
                    
                        if branchJet.At(j) in associated_jets:
                            images_partons.append(jet_im)
                    
                else:
                    
                    for j in range(branchJet.GetEntries()):
                        
                        #building 3 dimensional image
                        jet_plus_im = []
                        jet_minus_im = []
                        jet_neutral_im = []
                        
                        particles = branchJet.At(j).Constituents
                        jet_Eta = branchJet.At(j).Eta
                        jet_Phi = branchJet.At(j).Phi
                        
                        for p in range(particles.GetEntries()):
                            
                            if(type(particles.At(p)) == ROOT.GenParticle ):
                                four_mom_p = particles.At(p).P4()
                                pt = four_mom_p.Pt()
                                charge = particles.At(p).Charge
                                
                                if charge == -1 :
                                    jet_minus_im.append([jet_Eta-particles.At(p).Eta, jet_Phi-particles.At(p).Phi, pt ])
                                elif charge == 0 :
                                    jet_neutral_im.append([jet_Eta-particles.At(p).Eta, jet_Phi-particles.At(p).Phi, pt ])
                                elif charge == 1 :
                                    jet_plus_im.append([jet_Eta-particles.At(p).Eta, jet_Phi-particles.At(p).Phi, pt ])
                            
                            if(type(particles.At(p)) == ROOT.Track ):
                                four_mom_p = particles.At(p).P4()
                                pt = four_mom_p.Pt()
                                charge = particles.At(p).Charge
                                
                                if charge == -1 :
                                    jet_minus_im.append([jet_Eta-particles.At(p).Eta, jet_Phi-particles.At(p).Phi, pt ])
                            
                                elif charge == 1 :
                                    jet_plus_im.append([jet_Eta-particles.At(p).Eta, jet_Phi-particles.At(p).Phi, pt ])
                            
                            if(type(particles.At(p)) == ROOT.Tower ):
                                four_mom_p = particles.At(p).P4()
                                pt = four_mom_p.Pt()
                                jet_neutral_im.append([jet_Eta-particles.At(p).Eta, jet_Phi-particles.At(p).Phi, pt ])
                    
                    
                        if branchJet.At(j) in associated_jets:
                            images_partons.append([jet_minus_im, jet_neutral_im, jet_plus_im])

        #return
        return images_partons

    def create_jet_image(image_vector, bins=100, eta=[-0.8, 0.8], phi=[-0.8, 0.8]):
        
        etas = np.linspace(eta[0], eta[1], bins)
        phis = np.linspace(phi[0], phi[1], bins)

        im_to_plt = []
        count = 0
        for jet_im in tqdm(image_vector):
                
            im = np.zeros((bins,bins))
            for i in range(bins-1):
                for j in range(bins-1):
                    eta_inf = etas[i]
                    eta_sup = etas[i+1]
                    phi_inf = phis[j]
                    phi_sup = phis[j+1]
                    for el in jet_im:
                        if (el[0] > eta_inf) & (el[0] < eta_sup) & (el[1] > phi_inf) & (el[1] < phi_sup):
                            im[i,j] += el[2]
            im_to_plt.append(im)
            count += 1
            
        return np.array(im_to_plt)
    
    def create_jet_3d_image(threed_image_vector, bins=100, eta=[-0.8, 0.8], phi=[-0.8, 0.8]):
        
        etas = np.linspace(eta[0], eta[1], bins)
        phis = np.linspace(phi[0], phi[1], bins)

        im_to_plt = []
        for jet_im in tqdm(threed_image_vector):
                
            im_minus = np.zeros((bins,bins))
            im_plus = np.zeros((bins,bins))
            im_neutral = np.zeros((bins,bins))
            
            #shape = (3, bins, bins)
            im = np.array([im_minus, im_neutral, im_plus])
            
            for dim, charged_part in enumerate(jet_im):
                for i in range(bins-1):
                    for j in range(bins-1):
                        eta_inf = etas[i]
                        eta_sup = etas[i+1]
                        phi_inf = phis[j]
                        phi_sup = phis[j+1]
                        for el in charged_part:
                            if (el[0] > eta_inf) & (el[0] < eta_sup) & (el[1] > phi_inf) & (el[1] < phi_sup):
                                im[dim,i,j] += el[2]
            im_to_plt.append(im)
            
        return np.array(im_to_plt)
        










