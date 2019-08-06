import numpy as np 
import argparse 
from Creators.variable_creator import variable_creator
from Creators.image_creator import image_creator
from Utils.plot_image import plot_image
from Utils.generic_utils import mkdir_p
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True,
                    help="Full numpy file")
parser.add_argument('-a', '--att', type=str, required=True,
                    help="to add numpy file")

parser.add_argument('-s', '--save_path', type=str, required=True,
                    help="path to save, specify .npy")

parser.add_argument('-t', '--type', type=str, required=True,
                    help="type either image or variable")

parser.add_argument('-q', '--parton', type=str, required=False,
                    help="q for quarks, g for gluons")

parser.add_argument('-e', '--eta_range', type=list, required=False,
                    help="range of eta in bins")

parser.add_argument('-p', '--phi_range', type=list, required=False,
                    help="range of phi in bins")

args = parser.parse_args()

#importing libraries to read Delphes tree
ROOT.gSystem.Load("/Users/bcoder/MG5_aMC_v2_6_6/Delphes/libDelphes.so")
ROOT.gSystem.Load("/Users/bcoder/MG5_aMC_v2_6_6/Delphes/libDelphes.so")
ROOT.gSystem.Load("/Users/bcoder/MG5_aMC_v2_6_6/Delphes/external/libExRootAnalysis.so")
ROOT.gInterpreter.Declare('#include "/Users/bcoder/MG5_aMC_v2_6_6/ExRootAnalysis/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gInterpreter.Declare('#include "/Users/bcoder/MG5_aMC_v2_6_6/Delphes/classes/SortableObject.h"')
ROOT.gInterpreter.Declare('#include "/Users/bcoder/MG5_aMC_v2_6_6/Delphes/classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare('#include "/Users/bcoder/MG5_aMC_v2_6_6/ExRootAnalysis/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gInterpreter.Declare('#include "/Users/bcoder/MG5_aMC_v2_6_6/ExRootAnalysis/ExRootAnalysis/ExRootTask.h"')
ROOT.gInterpreter.Declare('#include "/Users/bcoder/MG5_aMC_v2_6_6/Delphes/classes/SortableObject.h"')
ROOT.gInterpreter.Declare('#include "/Users/bcoder/MG5_aMC_v2_6_6/Delphes/classes/DelphesClasses.h"')

#improting original numpy dataset
full_dataset = np.load(args.file)

if args.type == "image":
    
    #can only concatenate images of same dimension, forcing on the existing one 
    #the binning
    bin = full_dataset.shape[1]
    
    if not args.eta_range:
        etas = [-0.9,0.9]
    else:
        etas = args.eta_range
        
    if not args.phi_range:
        phis = [-0.9, 0.9]
    else:
        phis = args.phi_range
    
    append_eta_phi_pt = image_creator.Jet_Pt_dist([args.att], parton_to_match = args.parton)
    
    append_im = image_creator.create_jet_image(append_eta_phi_pt, bins=bin, eta=etas, phi=phis)
    
    print("Shapes: ")
    print(len(append_eta_phi_pt))
    print(append_im.shape)
    print(full_dataset.shape)
    final = np.concatenate((full_dataset, append_im), axis = 0)
    
    np.save(args.save_path, final)
        
if args.type == "variable":
    
    append_var = variable_creator.create_jet_array([args.att], parton_to_match = args.parton)
    final = np.concatenate((full_dataset, append_var), axis = 0)
    np.save(args.save_path, final)

