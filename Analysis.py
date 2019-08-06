import numpy as np
import math as mt
from Creators.variable_creator import variable_creator
from Creators.image_creator import image_creator
from Utils.plot_image import plot_image
from Utils.generic_utils import mkdir_p
import ROOT 

#creating saving repo

save_path = "./data_general"
mkdir_p(save_path)

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

#path_to_gluons = ['/Users/bcoder/Bonesini_qq_gg/200Gev/data_200/5th_run/pp_gg_200.root']
#path_to_quarks = ['/Users/bcoder/Bonesini_qq_gg/200Gev/data_200/5th_run/pp_qq_200.root']

path_to_gluons = ['/Users/bcoder/Bonesini_qq_gg/200Gev/data_200/4nd_run/pp_gg_200.root',
                  '/Users/bcoder/Bonesini_qq_gg/200Gev/data_200/5rd_run/pp_gg_200.root',
                  '/Users/bcoder/Bonesini_qq_gg/200Gev/data_200/6th_run/pp_gg_200.root']

path_to_quarks = ['/Users/bcoder/Bonesini_qq_gg/200Gev/data_200/4nd_run/pp_qq_200.root',
                  '/Users/bcoder/Bonesini_qq_gg/200Gev/data_200/5rd_run/pp_qq_200.root',
                  '/Users/bcoder/Bonesini_qq_gg/200Gev/data_200/6th_run/pp_qq_200.root']
                  
"""
#creating variables
#gluon_var = variable_creator.create_jet_array(path_to_gluons, parton_to_match = 'g')
#quark_var = variable_creator.create_jet_array(path_to_quarks, parton_to_match = 'q')

#np.save(save_path+"/gluon_var.npy", gluon_var)
#np.save(save_path+"/quark_var.npy", quark_var)

#creating images
gluon_eta_phi_pt = image_creator.Jet_Pt_dist(path_to_gluons, parton_to_match = 'g')
gluon_im = image_creator.create_jet_image(gluon_eta_phi_pt, bins=130, eta=[-0.9, 0.9], phi=[-0.9, 0.9])
np.save(save_path+"/gluon_im.npy", gluon_im)

quark_eta_phi_pt = image_creator.Jet_Pt_dist(path_to_quarks, parton_to_match = 'q')
quark_im = image_creator.create_jet_image(quark_eta_phi_pt, bins=130, eta=[-0.9, 0.9], phi=[-0.9, 0.9])
np.save(save_path+"/quark_im.npy", quark_im)


plot_image(gluon_im, Save = True, title = "Gluon Pt density")
plot_image(quark_im, Save = True, title = "Quark Pt density")
"""

#creating 3d images
quark_eta_phi_pt_3d = image_creator.Jet_Pt_dist(path_to_quarks, parton_to_match = 'q', three_d=True)
quark_3d_im = image_creator.create_jet_3d_image(quark_eta_phi_pt_3d, bins=130, eta=[-0.9, 0.9], phi=[-0.9, 0.9])
np.save(save_path+"/quarks_3d_im.npy", quark_3d_im)

gluon_eta_phi_pt_3d = image_creator.Jet_Pt_dist(path_to_gluons, parton_to_match = 'g', three_d=True)
gluon_3d_im = image_creator.create_jet_3d_image(gluon_eta_phi_pt_3d, bins=130, eta=[-0.9, 0.9], phi=[-0.9, 0.9])
np.save(save_path+"/gluons_3d_im.npy", gluon_3d_im)




