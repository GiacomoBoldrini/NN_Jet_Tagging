import math as mt
import numpy as np
import pandas as pd
import ROOT
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from Utils.Jet_parton_match import Jet_parton_match

class variable_creator():
    
    def create_jet_array(data_path, parton_to_match = 'q'):
        
        chain = ROOT.TChain("Delphes")
        for file in data_path:
            chain.Add(file)
        
        treeReader = ROOT.ExRootTreeReader(chain)
        treeReader = ROOT.ExRootTreeReader(chain)
        numberOfEntries = treeReader.GetEntries()
        
        branchJet = treeReader.UseBranch("Jet")
        branchFatJet = treeReader.UseBranch("FatJet")
        branchMuon = treeReader.UseBranch("Muon")
        branchElectron = treeReader.UseBranch("Electron")
        branchMET = treeReader.UseBranch("MissingET")
        branchPhoton = treeReader.UseBranch("Photon")
        branchParticle = treeReader.UseBranch("Particle");
        branchEFlowTrack = treeReader.UseBranch("EFlowTrack");
        branchEFlowTower = treeReader.UseBranch("EFlowTower");
        branchEFlowMuon = treeReader.UseBranch("EFlowMuon");
        branchEFlowPhoton = treeReader.UseBranch("EFlowPhoton");
        branchEFlowNeutralHadron = treeReader.UseBranch("EFlowNeutralHadron");
        branchTower = treeReader.UseBranch("Tower")

        jet_array = []
        
        print("Total entries: ",numberOfEntries)
        for entry in tqdm(range(0, numberOfEntries)):
            #for entry in tqdm(range(0, 100)):
            
            
            treeReader.ReadEntry(entry)
            number_of_jets = branchJet.GetEntries()
            #print("Number of jets: ", number_of_jets)
            
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
                
                for j in associated_jets:
                    j_att = []
                    j_att.append(j.PT) #T mom
                    j_att.append(j.Eta) #Eta
                    j_att.append(j.Phi) #Phi
                    j_att.append(branchJet.GetEntries()) #number of jets
                    j_att.append(j.Mass) #invariant mass
                    j_att.append(j.Mass/j.PT) #rateo
                    j_att.append(j.NCharged) #charged multiplicity
                    j_att.append(j.NNeutrals) #neutral multiplicity
                    j_att.append(j.EhadOverEem) #Rateo of energy in ECAL and HCAL
                    j_att.append(j.Tau[0]) #starting with N-Subjettiness
                    if j.Tau[0] > 0:
                        j_att.append(j.Tau[1]/j.Tau[0])
                    else:
                        j_att.append(0)
                    
                    j_att.append(j.Tau[1])
                    if j.Tau[1] > 0:
                        j_att.append(j.Tau[2]/j.Tau[1])
                    else:
                        j_att.append(0)
                    
                    j_att.append(j.Tau[2])
                    if j.Tau[2] > 0:
                        j_att.append(j.Tau[3]/j.Tau[2])
                    else:
                        j_att.append(0)
                    
                    j_att.append(j.Tau[3])
                    if j.Tau[3] > 0:
                        j_att.append(j.Tau[4]/j.Tau[3])
                    else:
                        j_att.append(0)
                    
                    
                    jet_array.append(j_att)

        return jet_array
