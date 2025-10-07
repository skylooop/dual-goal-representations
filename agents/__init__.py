from agents.crl.original import CRLAgent
from agents.crl.byol import CRLBYOLAgent
from agents.crl.dual import CRLDualAgent
from agents.crl.tra import CRLTRAAgent
from agents.crl.vib import CRLVIBAgent
from agents.crl.vip import CRLVIPAgent
from agents.gcfbc.original import GCFBCAgent
from agents.gcfbc.byol import GCFBCBYOLAgent
from agents.gcfbc.dual import GCFBCDualAgent
from agents.gcfbc.tra import GCFBCTRAAgent
from agents.gcfbc.vib import GCFBCVIBAgent
from agents.gcfbc.vip import GCFBCVIPAgent
from agents.gcivl.original import GCIVLAgent
from agents.gcivl.pixel.dual import GCIVLVisualDualAgent
from agents.gcivl.pixel.byol import GCIVLVisualBYOLAgent
from agents.gcivl.pixel.tra import GCIVLVisualTRAAgent
from agents.gcivl.pixel.vib import GCIVLVisualVIBAgent
from agents.gcivl.pixel.vip import GCIVLVisualVIPAgent
from agents.gcivl.state.dual import GCIVLDualAgent
from agents.gcivl.state.byol import GCIVLBYOLAgent
from agents.gcivl.state.tra import GCIVLTRAAgent
from agents.gcivl.state.vib import GCIVLVIBAgent
from agents.gcivl.state.vip import GCIVLVIPAgent

agents = dict(
    crl=CRLAgent,
    gcfbc=GCFBCAgent,
    gcivl=GCIVLAgent,
    # Dual representation agents.
    gcivl_dual=GCIVLDualAgent,
    gcivl_dual_vis=GCIVLVisualDualAgent,
    crl_dual=CRLDualAgent,
    gcfbc_dual=GCFBCDualAgent,
    # VIB agents.
    gcivl_vib=GCIVLVIBAgent,
    gcivl_vib_vis=GCIVLVisualVIBAgent,
    crl_vib=CRLVIBAgent,
    gcfbc_vib=GCFBCVIBAgent,
    # TRA agents.
    gcivl_tra=GCIVLTRAAgent,
    gcivl_tra_vis=GCIVLVisualTRAAgent,
    crl_tra=CRLTRAAgent,
    gcfbc_tra=GCFBCTRAAgent,
    # BYOL agents.
    gcivl_byol=GCIVLBYOLAgent,
    gcivl_byol_vis=GCIVLVisualBYOLAgent,
    crl_byol=CRLBYOLAgent,
    gcfbc_byol=GCFBCBYOLAgent,
    # VIP agents.
    gcivl_vip=GCIVLVIPAgent,
    gcivl_vip_vis=GCIVLVisualVIPAgent,
    crl_vip=CRLVIPAgent,
    gcfbc_vip=GCFBCVIPAgent,
)
