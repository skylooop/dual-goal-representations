# GCIVL original
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99

# GCIVL + dual
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# GCIVL + BYOL
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99

# GCIVL + TRA
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99

# GCIVL + VIB
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.003 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.003 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.003 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99

# GCIVL + VIP
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcivl/state/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99

# CRL original
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/crl/original.py --agent.alpha=0.03 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/crl/original.py --agent.alpha=0.03 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/crl/original.py --agent.alpha=0.1 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/crl/original.py --agent.alpha=0.1 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/crl/original.py --agent.alpha=0.1 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/crl/original.py --agent.alpha=0.1 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/crl/original.py --agent.alpha=0.1 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/crl/original.py --agent.alpha=0.3 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/crl/original.py --agent.alpha=3.0 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/crl/original.py --agent.alpha=3.0 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/crl/original.py --agent.alpha=3.0 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/crl/original.py --agent.alpha=3.0 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/crl/original.py --agent.alpha=3.0 --agent.discount=0.99

# CRL + dual
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/crl/dual.py --agent.alpha=0.03 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/crl/dual.py --agent.alpha=0.03 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/crl/dual.py --agent.alpha=0.1 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/crl/dual.py --agent.alpha=0.1 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/crl/dual.py --agent.alpha=0.1 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/crl/dual.py --agent.alpha=0.1 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/crl/dual.py --agent.alpha=0.1 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/crl/dual.py --agent.alpha=0.3 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/crl/dual.py --agent.alpha=3.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/crl/dual.py --agent.alpha=3.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/crl/dual.py --agent.alpha=3.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/crl/dual.py --agent.alpha=3.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/crl/dual.py --agent.alpha=3.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# CRL + BYOL
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/crl/byol.py --agent.alpha=0.03 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/crl/byol.py --agent.alpha=0.03 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/crl/byol.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/crl/byol.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/crl/byol.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/crl/byol.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/crl/byol.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/crl/byol.py --agent.alpha=0.3 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/crl/byol.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/crl/byol.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/crl/byol.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/crl/byol.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/crl/byol.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99

# CRL + TRA
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/crl/tra.py --agent.alpha=0.03 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/crl/tra.py --agent.alpha=0.03 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/crl/tra.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/crl/tra.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/crl/tra.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/crl/tra.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/crl/tra.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/crl/tra.py --agent.alpha=0.3 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/crl/tra.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/crl/tra.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/crl/tra.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/crl/tra.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/crl/tra.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99

# CRL + VIB
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/crl/vib.py --agent.alpha=0.03 --agent.beta=0.003 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/crl/vib.py --agent.alpha=0.03 --agent.beta=0.003 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/crl/vib.py --agent.alpha=0.1 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/crl/vib.py --agent.alpha=0.1 --agent.beta=0.003 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/crl/vib.py --agent.alpha=0.1 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/crl/vib.py --agent.alpha=0.1 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/crl/vib.py --agent.alpha=0.1 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/crl/vib.py --agent.alpha=0.3 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/crl/vib.py --agent.alpha=3.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/crl/vib.py --agent.alpha=3.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/crl/vib.py --agent.alpha=3.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/crl/vib.py --agent.alpha=3.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/crl/vib.py --agent.alpha=3.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99

# CRL + VIP
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/crl/vip.py --agent.alpha=0.03 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/crl/vip.py --agent.alpha=0.03 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/crl/vip.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/crl/vip.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/crl/vip.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/crl/vip.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/crl/vip.py --agent.alpha=0.1 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/crl/vip.py --agent.alpha=0.3 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/crl/vip.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/crl/vip.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/crl/vip.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/crl/vip.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/crl/vip.py --agent.alpha=3.0 --agent.goalrep_dim=256 --agent.discount=0.99

# GCFBC original
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcfbc/original.py --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcfbc/original.py --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcfbc/original.py --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcfbc/original.py --agent.discount=0.99

# GCFBC + dual
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcfbc/dual.py --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99

# GCFBC + BYOL
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcfbc/byol.py --agent.goalrep_dim=256 --agent.discount=0.99

# GCFBC + TRA
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcfbc/tra.py --agent.goalrep_dim=256 --agent.discount=0.99

# GCFBC + VIB
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.003 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.003 --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.003 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcfbc/vib.py --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99

# GCFBC + VIP
python main.py --env_name=pointmaze-medium-navigate-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=pointmaze-large-navigate-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=64 --agent.discount=0.99
python main.py --env_name=antmaze-medium-navigate-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-large-navigate-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=antmaze-giant-navigate-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-single-play-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=cube-double-play-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=scene-play-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-3x3-play-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.99
python main.py --env_name=puzzle-4x4-play-v0 --agent=agents/gcfbc/vip.py --agent.goalrep_dim=256 --agent.discount=0.99

# Visual GCIVL original
python main.py --env_name=visual-antmaze-medium-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-antmaze-large-navigate-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-double-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-scene-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-3x3-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-4x4-play-v0 --agent=agents/gcivl/original.py --agent.alpha=10.0 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15

# Visual GCIVL + dual
python main.py --env_name=visual-antmaze-medium-navigate-v0 --agent=agents/gcivl/pixel/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-antmaze-large-navigate-v0 --agent=agents/gcivl/pixel/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.9 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/gcivl/pixel/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-double-play-v0 --agent=agents/gcivl/pixel/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-scene-play-v0 --agent=agents/gcivl/pixel/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-3x3-play-v0 --agent=agents/gcivl/pixel/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-4x4-play-v0 --agent=agents/gcivl/pixel/dual.py --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15

# Visual GCIVL + BYOL
python main.py --env_name=visual-antmaze-medium-navigate-v0 --agent=agents/gcivl/pixel/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-antmaze-large-navigate-v0 --agent=agents/gcivl/pixel/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/gcivl/pixel/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-double-play-v0 --agent=agents/gcivl/pixel/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-scene-play-v0 --agent=agents/gcivl/pixel/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-3x3-play-v0 --agent=agents/gcivl/pixel/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-4x4-play-v0 --agent=agents/gcivl/pixel/byol.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15

# Visual GCIVL + TRA
python main.py --env_name=visual-antmaze-medium-navigate-v0 --agent=agents/gcivl/pixel/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-antmaze-large-navigate-v0 --agent=agents/gcivl/pixel/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/gcivl/pixel/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-double-play-v0 --agent=agents/gcivl/pixel/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-scene-play-v0 --agent=agents/gcivl/pixel/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-3x3-play-v0 --agent=agents/gcivl/pixel/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-4x4-play-v0 --agent=agents/gcivl/pixel/tra.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15

# Visual GCIVL + VIB
python main.py --env_name=visual-antmaze-medium-navigate-v0 --agent=agents/gcivl/pixel/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-antmaze-large-navigate-v0 --agent=agents/gcivl/pixel/vib.py --agent.alpha=10.0 --agent.beta=0.003 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/gcivl/pixel/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-double-play-v0 --agent=agents/gcivl/pixel/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-scene-play-v0 --agent=agents/gcivl/pixel/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-3x3-play-v0 --agent=agents/gcivl/pixel/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-4x4-play-v0 --agent=agents/gcivl/pixel/vib.py --agent.alpha=10.0 --agent.beta=0.001 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15

# Visual GCIVL + VIP
python main.py --env_name=visual-antmaze-medium-navigate-v0 --agent=agents/gcivl/pixel/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-antmaze-large-navigate-v0 --agent=agents/gcivl/pixel/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-single-play-v0 --agent=agents/gcivl/pixel/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-cube-double-play-v0 --agent=agents/gcivl/pixel/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-scene-play-v0 --agent=agents/gcivl/pixel/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-3x3-play-v0 --agent=agents/gcivl/pixel/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15
python main.py --env_name=visual-puzzle-4x4-play-v0 --agent=agents/gcivl/pixel/vip.py --agent.alpha=10.0 --agent.goalrep_dim=256 --agent.discount=0.99 --agent.p_aug=0.5 --train_steps=500000 --agent.batch_size=256 --agent.encoder=impala_small --eval_episodes=15