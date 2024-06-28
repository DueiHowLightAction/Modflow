from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import flopy
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import os
import sys
import matplotlib.gridspec as gridspec
import pandas as pd

print(sys.version)
print(f"flopy version: {flopy.__version__}")

path = os.path.join("workspace")
path = os.path.join("sensitivity_study\sfe0")
path = os.path.join("sensitivity_study\layer 2\sfe2")

##### Creating the MODFLOW Model #####
temp_dir = TemporaryDirectory()
workspace = Path(r"D:\NTU_Stride-C\codes\modflow\workspace")
#workspace = Path(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\sfe0")
workspace = Path(r'D:\NTU_Stride-C\codes\modflow\sensitivity_study\layer 2\sfe3_nn=80')
name = "SUB-Yulin-5_0"
mf = flopy.modflow.Modflow(name, exe_name="mf2005", model_ws=workspace)

# DIS
Lx = 25000        #25870
Ly = 2500
nlay = 4
ncol = 200
nrow = 20
ztop = 0 #[0, -100, -150]
#zbot = [-1320.0, -1688.0, -2500.0]
zbot = np.zeros((nlay, nrow, ncol), dtype = np.float16)
zbot[0, :, :] = -300
for i in range(ncol):
    zbot[1, :, i] = -1320 - (1800-1320)/ncol*i
    zbot[2, :, i] = -1688 - (2150-1688)/ncol*i
zbot[3, :, :] = -2500
delr = Lx / ncol
delc = Ly / nrow
botm = zbot

nper = 11
nper = 7       
perlen = np.concatenate(([1], np.ones(nper-1) * 365.3))         # 10年 
nstp = np.concatenate(([1], np.ones(nper-1, dtype=int) * 6))
steady = np.concatenate(([True], np.full(nper-1, False, dtype=bool)))
tsmult = np.concatenate(([1], np.ones(nper-1) * 1.3))
lenuni = 0

dis = flopy.modflow.ModflowDis(
    mf, 
    nlay, 
    nrow, 
    ncol, 
    delr=delr, 
    delc=delc, 
    top=ztop, 
    botm=botm,
    nper=nper,
    perlen=perlen,
    nstp=nstp,
    steady=steady,
    tsmult=tsmult,
    lenuni=lenuni,
)

# BAS BASic
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)    #If IBOUND< 0, constant head. = 0, is no flow. > 0, is variable head.
ibound[:, :, :] = 2
ibound[:, :, int(ncol-1)] = -1
ibound[:, :, 0] = -1
strt = 10.0*np.ones((nlay, nrow, ncol),dtype=np.float32)
strt[:, :, ncol-1] = 4
strt[:, :, 0] = -7
#for i in range(ncol):
#    for j in range(nlay-1):

#        strt[j, :, i] = (-zbot[j, 0, i]-0)/2
#         strt[j+1, :, i] = (-zbot[j, 0, i]-0)/2

#strt[0, :, :] = zbot[0, :, 0]
#strt[1, :, :] = zbot[1, :, 0]
#strt[2, :, :] = zbot[2, :, 0]
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

# BCF   Block Centered Flow Package
# https://flopy.readthedocs.io/en/latest/source/flopy.modflow.mfbcf.html
laycon = [1, 0, 0, 0]      # laycon = Ltype, 1 = unconfined, anothers = confined
sf1 = [0.0077, 0.0077,  6.56e-5, 2.62e-4]       # specific storage
# tran if Ltype = 0, 2
# hy if Ltype = 1, 3 
hy = 7.34                             # hydraulic conductivity
tran = np.ones((nlay, nrow, ncol), dtype='f')
tran[1, :, :] = 0.5
tran[1, 0, :] = 1000     # 第一橫排全部
tran[1, :, 0] = 1000
tran[1, :, -1] = 1000
#vcont = [3e-6, 3e-6] 
vcont = [1e-6, 3e-6, 3e-6]   # 兩層間的leakence (conductance)，總共三層所以兩個值
#vcont = [1e-5, 3e-6, 3e-6]
tran[2, :, :] = 1000       # only laycon = 0 or 2
tran[3, :, :] = 1000

bcf = flopy.modflow.ModflowBcf(
    mf, 
    laycon=laycon,
    hdry=0,
    wetfct=0,       # factor used when cell is converted from dry to wet (default is 0.1)
    iwetit=0,       # iteration interval in wetting/drying algorithm (default is 1)
    tran=tran,
    hy=hy,
    sf1=sf1, 
    vcont=vcont
)

# WEL WELL 
pumping_rate = -1.53169e5
recharging_rate = 100
wel_sp = np.zeros((1, 4))
wel_sp[0, 1] = nrow/2-1
wel_sp[0, 2] = ncol/2-1
wel_sp[0, 3] = pumping_rate
stress_period_data = {i: wel_sp for i in range(1, nper)}
wel = flopy.modflow.ModflowWel(
    mf, 
    stress_period_data=stress_period_data
)

# rch Recharge
nrchop = 1
rech = 1.4e-2
#rech = 0.1368  #11.81e9/365/(1074.0e6+1291.0e6)
#rech = 0.1368 # m/day
rch = flopy.modflow.ModflowRch(
    mf, 
    rech = rech,
    nrchop = nrchop,
)

# SUB
# https://flopy.readthedocs.io/en/latest/source/flopy.modflow.mfsub.html
isuboc = 1
nndb = 4        # num of no-delay interbed
ndb = 2         # num of delay interbed
nmz = 1         # num of zones needed to define properties (hydraulic conduction, elastic/inelastic specific storage )
nn = 20
nn = 40
nn = 80          
ac1 = 0 
ac2 = 0.2
itmin = 5
idsave = -1
idrest = -1
#for i in range(nlay):
ln = [0, 1, 2, 3]          # model layer assignments 這邊輸入要比實際上-1，ldn也是
ldn = [0, 1]
rnb = [7.635, 17.718]
hc = [6.2, 6.2, 6.2, 6.2]       # preconsolidation head (stress)  
#
sfe0 = 1.3e-4       # 宏崙井最小的
sfe00 = 4.6e-4
sfe01 = 5.9e-4
sfe1 = 1.14286e-3   # (0.00081+0.00059+0.00192+0.00022+0.00046+0.00013+0.00387)/7
sfe2 = 2.53e-3      # 洪秋香分區的值            
sfe3 = 5.0e-3       # 敏感度測試             
sfe = [sfe01, sfe3, 1.5e-3, 4.2e-3]                    # elastic skeletal storage coefficient
sfv = np.ones((nlay, nrow, ncol), dtype='e')            # inelastic skeletal storage coefficient
sfv2 = 6.515e-3     #(0.00140+0.00684+0.00167+0.00901+0.00217+0.01800)/6
sfv[0, :, :] = sfv2
sfv[1, :, :] = 9.12e-3
sfv[1, :, :] = sfv2
sfv[2, :, :] = 7.58e-3
sfv[3, :, :] = 0.01824
com = 0
#
dz = [5.894, 5.080]            # equivalent thickness for a system of delay interbeds
dz = [20, 5.0]
dz = [40, 5.0]
dz = [40, 80]
dz = [75, 80]
#dz = [40, 400]
nz = 1
dstart = -7                     # starting head of delay interbeds
dhc = 6.2                       #(12.07+8.93+5.76+4.37+2.98+4.09+5.2)/7
#dhc = -7                       # starting preconsolidation head of delay interbeds
dcom = 0                        # starting compaction of delay interbed
ids15 = [0, 41, 0, 42, 0, 42, 0, 41, 0, 43, 0, 43]
ids16 = [1-1, nper-1, 6-1, 6-1, -1, 2, -1, 2, -1, 2, -1, -1, -1, -1, -1, -1, 2]
sub = flopy.modflow.ModflowSub(
    mf,
    isuboc=isuboc,
    nndb=nndb,
    ndb=ndb,
    nmz=nmz,
    nn=nn, 
    ac1=ac1,
    ac2=ac2,
    itmin=itmin,
    idsave=idsave,
    idrest=idrest,
    ln=ln,
    ldn=ldn,
    rnb=rnb,
    hc=hc,
    sfe=sfe,
    sfv=sfv,
    com=com,
    dz=dz,
    nz=nz,
    dstart=dstart,
    dhc=dhc,
    dcom=dcom,
    ids15=ids15,
    ids16=ids16,
)

# SIP (Strongly Implicit Procedure)
mxiter = 120
nparm = 5
accl = 1
hclose = 1.0e-4
ipcalc = 1
wseed = 0
iprsip = 5
sip = flopy.modflow.ModflowSip(
    mf,
    mxiter=mxiter,
    nparm=nparm,
    accl=accl,
    hclose=hclose,
    ipcalc=ipcalc,
    wseed=wseed,
    iprsip=iprsip,
)

# OC Output Control
stress_period_data = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        stress_period_data[(kper, kstp)] = [
            "save head",
            "save drawdown",
            "save budget",
            "print head",
            "print budget",
        ]
oc = flopy.modflow.ModflowOc(
    mf, stress_period_data=stress_period_data, compact=True
)
#oc = flopy.modflow.ModflowOc.load(r'D:\NTU_Stride-C\codes\modflow\SUB-Yulin-2\SUB-Yulin-2-2.oc', mf)

# Write the model input file
mf.write_input() 

# 
m = flopy.modflow.Modflow.load(name, model_ws=workspace)
#m.change_model_ws(workspace)
chk = mf.check()

# Run the Model
success, mfoutput = mf.run_model(silent=True, pause=False)
assert success, "MODFLOW did not terminate normally."


##### Post-Processing the Results #####

# Then we can use the plot_grid() method to draw the grid
# The return value for this function is a matplotlib LineCollection object,
# which could be manipulated (or used) later if necessary.

# plotting B.C. (iBound) (well) 

#ax = fig.add_subplot(1, 1, 1)

#xsect = flopy.plot.PlotCrossSection(model=mf, line={"Row": 0})

#t = ax.set_title("XZ Cross-Section with Boundary Conditions")

workspace1 = Path(r"D:\NTU_Stride-C\codes\modflow\workspace\SUB-Yulin-5")
storespace = Path(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\SUB-Yulin-5")

# 地層下陷量的二進位檔轉ASCII
sobj = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\workspace\SUB-Yulin-5.subsidence.hds",
                   text="SUBSIDENCE")
sobj1 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\workspace\SUB-Yulin-5.total_comp.hds",
                   text="COMPACTION")
sobj2 = bf.HeadFile(workspace1.with_suffix(".hds"),)

times = sobj.get_times()
print("times steps are: ", times)
subs = sobj.get_data(totim=times[1])
subs_t = np.zeros((len(times), ncol))
comp1_t = np.zeros((len(times), ncol))
comp2_t = np.zeros((len(times), ncol))
comp3_t = np.zeros((len(times), ncol))
comp4_t = np.zeros((len(times), ncol))

for i in range(len(times)):
    for j in range(ncol):
        subs_t[i, j] = sobj.get_data(totim=times[i])[0,0,j]
        comp1_t[i, j] = sobj1.get_data(totim=times[i])[0,0,j]
        comp2_t[i, j] = sobj1.get_data(totim=times[i])[1,0,j]
        comp3_t[i, j] = sobj1.get_data(totim=times[i])[2,0,j]
        comp4_t[i, j] = sobj1.get_data(totim=times[i])[3,0,j]

       
comp = sobj1.get_data()
head = sobj2.get_data(totim=times[1])
# v_disp是3d array: [[[a, b, c]]]，需要取其中一個列
#np.savetxt(storespace.with_name("subs_t_2.txt"), subs_t[:])
#np.savetxt(storespace.with_name("compl1_t_2.txt"), comp1_t[:])
#np.savetxt(storespace.with_name("compl2_t_2.txt"), comp2_t[:])
#np.savetxt(storespace.with_name("compl3_t_2.txt"), comp3_t[:])
#np.savetxt(storespace.with_name("compl4_t_2.txt"), comp4_t[:])
#np.savetxt(workspace1.with_name("head.txt"), head[0])

# mesh
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)
ysect = flopy.plot.PlotCrossSection(model=mf, line={"Row": nrow/2-1})
#ysect.plot_fill_between(
#    [[[]]],
#    colors=["blue", "green"]
#)
linecollection = ysect.plot_grid()
t = plt.title("5th-Y Section with Boundary Conditions")
#plt.show()

# boundary condition
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)
ysect = flopy.plot.PlotCrossSection(model=mf, line={"Row": nrow/2-1})
patches = ysect.plot_bc("WEL", kper=1, color="pink")
patches = ysect.plot_ibound()
linecollection = ysect.plot_grid()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#t = plt.title("5th-Y Section with Boundary Conditions", fontsize=14) 
plt.show()

# col 1 的水頭剖面隨時間變化
plot_kper=[0, 1, 3, 6]        # 這樣設是為了下面畫well profile
mytimes = [times[plot_kper[0]], times[plot_kper[1]], times[plot_kper[2]], times[plot_kper[3]]]#, times[plot_kper[4]]]

# 找到所有時間水頭的最大最小值
global_min = np.inf     # 無窮大
global_max = -np.inf    # 負無窮大

for time in mytimes:
    head = sobj2.get_data(totim=time)
    global_min = min(global_min, head.min())
    global_max = max(global_max, head.max())

#plt.title("Heads of vertical section in row 1")
fig = plt.figure(figsize=(9, 9))

for iplot, time in enumerate(mytimes):
    print("*****Processing time: ", time)
    head = sobj2.get_data(totim=time)
    # Print statistics
    print("Head statistics")
    print("  min: ", head.min())
    print("  max: ", head.max())
    print("  std: ", head.std())

    ax = fig.add_subplot(len(mytimes)+1, 1, iplot + 1, aspect="equal")
    ax.set_title(f"day {int(time)}", fontsize=12)

    ysect = flopy.plot.PlotCrossSection(model=mf, line={"Row": (nrow/2-1)})
    pc = ysect.plot_array(head, head=head, alpha=0.5, ax=ax, vmin=global_min, vmax=global_max)
#    pc1 = ysect.contour_array(head, head=head, levels=20, alpha=0.5, ax=ax, vmin=global_min, vmax=global_max)#, levels=np.arange(1, 10, 1))
    patches = ysect.plot_ibound(head=head, ax=ax)
    linecollection = ysect.plot_grid(ax=ax)
    ax.set_xticklabels(ax.get_xticks(), fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    cb = plt.colorbar(pc, shrink=0.75, ax=ax)

    patches = ysect.plot_bc("WEL", kper=plot_kper[iplot], color="pink")
    patches = ysect.plot_ibound()
    linecollection = ysect.plot_grid()

plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.7) 
plt.show()


#fig = plt.figure(figsize=(10, 6))

#plt.plot(times, subs_t[:,50], "-o", label="Total Subsidence")
#plt.plot(times, comp1_t[:,50], "-o", label="Layer 1 compaction")
#plt.plot(times, comp2_t[:,50], "-o", label="Layer 2 compaction")
#plt.plot(times, comp3_t[:,50], "-o", label="layer 3 compaction")
#plt.xlabel("time (days)")
#plt.ylabel("subsidence(m)")
#plt.legend()
#
#plt.show()

# 畫出各層的壓密剖面和隨時間變化
extent = (delr / 2.0, Lx - delr / 2.0, Ly - delc / 2.0, delc / 2.0)

fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 2.5])
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])
ax4 = plt.subplot(gs[0, 3])
ax5 = plt.subplot(gs[1, :])

comp = sobj1.get_data(totim=times[1])
print(comp[0, :, :])

vmin = np.min(comp)
vmax = np.max(comp)

cax1 = ax1.contourf(comp[0, :, :], levels=20, extent=extent, vmin=vmin, vmax=vmax)
ax1.set_title("layer 1 compaction")

cax2 = ax2.contourf(comp[1, :, :], levels=20, extent=extent, vmin=vmin, vmax=vmax)
ax2.set_title("layer 2 compaction")

cax3 = ax3.contourf(comp[2, :, :], levels=20, extent=extent, vmin=vmin, vmax=vmax)
ax3.set_title("layer 3 compaction")

cax4 = ax4.contourf(comp[3, :, :], levels=20, extent=extent, vmin=vmin, vmax=vmax)
ax4.set_title("layer 4 compaction")

fig.colorbar(cax1, ax=[ax1, ax2, ax3, ax4], orientation='vertical', fraction=0.025, pad=0.04)

ax5.plot(times, -subs_t[:,49], "-o", label="Total Subsidence")
ax5.plot(times, -comp1_t[:,49], "-o", label="Layer 1 compaction")
ax5.plot(times, -comp2_t[:,49], "-o", label="Layer 2 compaction")
ax5.plot(times, -comp3_t[:,49], "-o", label="Layer 3 compaction")
ax5.plot(times, -comp4_t[:,49], "-o", label="Layer 4 compaction")
ax5.set_xlabel("time (days)")
ax5.set_ylabel("subsidence(m)")
ax5.legend()

# 調整圖之間的間距
#plt.tight_layout()

plt.show()


fig = plt.figure(figsize=(5, 15))

for iplot, time in enumerate(mytimes):
    comp = sobj1.get_data(totim=time)
    # Create a map for this time
    ax = fig.add_subplot(len(mytimes), 1, iplot + 1, aspect="equal")
    ax.set_title(f"stress period {iplot + 1}")
    pmv = flopy.plot.PlotMapView(model=mf, layer=0, ax=ax)
    qm = pmv.plot_ibound()
    lc = pmv.plot_grid()
    qm = pmv.plot_bc("WEL", alpha=0.5)
    if comp.min() != comp.max():
        cs = pmv.contour_array(comp, levels=20)
        plt.clabel(cs, inline=1, fontsize=10, fmt="%1.1f")

plt.show()

# 取GPS daily資料
sheet_name = "HLES宏崙"
file_path = "D:\modflow\Report\Subsidence Data2016-2021.xlsx"
df = pd.read_excel(file_path, sheet_name=sheet_name)

close_indices = []
time_len = len(df['day'])
day = df['day'].values
all_obs_subs = df['subsidence (mm)'].values
for itime in times:
    close_index = np.abs(day-itime).argmin()
    close_indices.append(close_index)
obs_subs = all_obs_subs[close_indices]

# 取300 m監測井資料
HLES = [0, 1.1, 1.9, 2.7, 3.3, 4.2, 4.7]#, 4.9, 0, 0, 0]
STES = [0, 1.3, 2.2, 3.7, 4.1, 5.3, 6.1]#, 6.3, 0, 0, 0]
TKES = [0, 1.5, 2.3, 3.2, 3.6, 5.4, 6.1]#, 5.8, 0, 0, 0]
for i in range(len(HLES)):
    HLES[i] = -HLES[i]*5/1.5/100
    STES[i] = -STES[i]*5/1.15/100
    TKES[i] = -TKES[i]*5/1.5/100


fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.plot(times, -comp1_t[:,49], "-o", label="Layer 1 Compaction")
ax1.plot(times, HLES, "-o", label="Well Observation")
ax1.set_title("Subsidence above 300 m")
ax1.set_xlabel("time (days)")
ax1.set_ylabel("subsidence(m)")
ax1.set_ylim(ymin=min(df["subsidence (mm)"])/1000)

ax2.plot(times, -comp1_t[:,49], "-o", label="Layer 1 Compaction")
ax2.plot(times, HLES, "-o", label="Well Observation")
ax2.set_title("Subsidence above 300 m")
ax2.plot(times, -subs_t[:,49], "-o", label="Total_subsidence")
ax2.plot(times, obs_subs/1000, "-o", label="GPS Observation")
ax2.set_title("Total Subsidence")
ax2.set_xlabel("time (days)")
ax2.set_ylabel("subsidence(m)")
ax2.set_ylim(ymin=min(df["subsidence (mm)"])/1000)

ax1.legend()
ax2.legend()

# 調整圖之間的間距
#plt.tight_layout()

plt.show()

try:
    temp_dir.cleanup()
except:
    # prevent windows permission error
    pass