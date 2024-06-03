from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import flopy
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import os
import sys
import matplotlib.gridspec as gridspec

print(sys.version)
print(f"flopy version: {flopy.__version__}")

path = os.path.join("workspace")

##### Creating the MODFLOW Model #####
temp_dir = TemporaryDirectory()
workspace = Path(r"D:\NTU_Stride-C\codes\modflow\workspace")
name = "SUB-Yulin-5"
mf = flopy.modflow.Modflow(name, exe_name="mf2005", model_ws=workspace)

# DIS
Lx = 25000        #25870
Ly = 500
nlay = 3
ncol = 200
nrow = 10
ztop = 0 #[0, -100, -150]
zbot = [-1320.0, -1688.0, -2500.0]
zbot = np.zeros((nlay, nrow, ncol), dtype = np.float16)
for i in range(ncol):
    zbot[0, :, i] = -1320 - (1800-1320)/ncol*i
    zbot[1, :, i] = -1688 - (2150-1688)/ncol*i
zbot[2, :, :] = -2500
delr = Lx / ncol
delc = Ly / nrow
botm = zbot

nper = 11       
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
for i in range(ncol):
    strt[0, :, i] = (-zbot[0, 0, i]-0)/2
    strt[1, :, i] = (-zbot[1, 0, i]-0)/2
    strt[2, :, i] = (-zbot[2, 0, i]-0)/2
#strt[0, :, :] = zbot[0, :, 0]
#strt[1, :, :] = zbot[1, :, 0]
#strt[2, :, :] = zbot[2, :, 0]
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

# BCF   Block Centered Flow Package
# https://flopy.readthedocs.io/en/latest/source/flopy.modflow.mfbcf.html
laycon = [1, 0, 0]      # laycon = Ltype, 1 = unconfined, anothers = confined
sf1 = [0.0077, 6.56e-5, 2.62e-4]       # specific storage
# tran if Ltype = 0, 2
# hy if Ltype = 1, 3 
hy = 7.34                             # hydraulic conductivity
tran = np.ones((nlay, nrow, ncol), dtype='f')
tran[1, :, :] = 0.5
tran[1, 0, :] = 1000     # 第一橫排全部
tran[1, :, 0] = 1000
tran[1, :, -1] = 1000
vcont = [3e-6, 3e-6]   # 兩層間的leakence (conductance)，總共三層所以兩個值
tran[2, :, :] = 1000       # only laycon = 0 or 2

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
wel_sp[0, 2] = ncol/2-1
wel_sp[0, 3] = pumping_rate
stress_period_data = {i: wel_sp for i in range(1, nper)}
wel = flopy.modflow.ModflowWel(
    mf, 
    stress_period_data=stress_period_data
)

# rch Recharge
nrchop = 1
rech = 0.1368 # m/day
rch = flopy.modflow.ModflowRch(
    mf, 
    rech = rech,
    nrchop = nrchop,
)

# SUB
# https://flopy.readthedocs.io/en/latest/source/flopy.modflow.mfsub.html
isuboc = 1
nndb = 3        # num of no-delay interbed
ndb = 2         # num of delay interbed
nmz = 1         # num of zones needed to define properties (hydraulic conduction, elastic/inelastic specific storage )
nn = 20          
ac1 = 0 
ac2 = 0.2
itmin = 5
idsave = -1
idrest = -1
ln = [0, 1, 2]          # model layer assignments 這邊輸入要比實際上-1，ldn也是
ldn = [0, 2]
rnb = [7.635, 17.718]
hc = [-7, -7, -7]       # preconsolidation head (stress) 
sfe = [2.1e-4, 1.5e-4, 4.2e-4]                         # elastic skeletal storage coefficient
sfv = np.ones((nlay, nrow, ncol), dtype='e')           # inelastic skeletal storage coefficient
sfv[0, :, :] = 9.12e-3
sfv[1, :, :] = 7.58e-3
sfv[2, :, :] = 0.01824
com = 0
dz = [5.894, 5.080]            # equivalent thickness for a system of delay interbeds
nz = 1
dstart = -7                     # starting head of delay interbeds
dhc = -7                        # starting preconsolidation head of delay interbeds
dcom = 0
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
m = flopy.modflow.Modflow.load("SUB-Yulin-5.nam", model_ws=path)
m.change_model_ws(workspace)
help(m.check)

chk = m.check()

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

# 地層下陷量的二進位檔轉ASCII
sobj = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\workspace\SUB-Yulin-5.subsidence.hds",
                   text="SUBSIDENCE")
sobj1 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\workspace\SUB-Yulin-5.total_comp.hds",
                   text="COMPACTION")
sobj2 = bf.HeadFile(workspace1.with_suffix(".hds"),)

times = sobj.get_times()
print(times)
subs = sobj.get_data(totim=times[1])
subs_t = np.zeros((len(times), ncol))
comp1_t = np.zeros((len(times), ncol))
comp2_t = np.zeros((len(times), ncol))
comp3_t = np.zeros((len(times), ncol))
print(len(times))
for i in range(len(times)):
    for j in range(ncol):
        subs_t[i, j] = sobj.get_data(totim=times[i])[0,0,j]
        comp1_t[i, j] = sobj1.get_data(totim=times[i])[0,0,j]
        comp2_t[i, j] = sobj1.get_data(totim=times[i])[1,0,j]
        comp3_t[i, j] = sobj1.get_data(totim=times[i])[2,0,j]

       
comp = sobj1.get_data()
head = sobj2.get_data(totim=times[1])
# v_disp是3d array: [[[a, b, c]]]，需要取其中一個列
np.savetxt(workspace1.with_name("subs.txt"), subs[0])
np.savetxt(workspace1.with_name("subs_t.txt"), subs_t[:])
np.savetxt(workspace1.with_name("compl1_t.txt"), comp1_t[:])
np.savetxt(workspace1.with_name("compl2_t.txt"), comp2_t[:])
np.savetxt(workspace1.with_name("compl1.txt"), comp[0])
np.savetxt(workspace1.with_name("compl2.txt"), comp[1])
np.savetxt(workspace1.with_name("compl3.txt"), comp[2])
np.savetxt(workspace1.with_name("head.txt"), head[0])


# col 1 的水頭剖面隨時間變化
#plt.title("Heads of vertical section in row 1")
fig = plt.figure(figsize=(15, 9))
plot_kper=[0, 1, 5, 10]        # 這樣設是為了下面畫well profile
mytimes = [times[plot_kper[0]], times[plot_kper[1]], times[plot_kper[2]], times[plot_kper[3]]]
for iplot, time in enumerate(mytimes):
    print("*****Processing time: ", time)
    head = sobj2.get_data(totim=time)
    # Print statistics
    print("Head statistics")
    print("  min: ", head.min())
    print("  max: ", head.max())
    print("  std: ", head.std())

    ax = fig.add_subplot(len(mytimes)+1, 1, iplot + 1, aspect="equal")
    ax.set_title(f"day {int(time)}")

    ysect = flopy.plot.PlotCrossSection(model=mf, line={"Row": 0})
    pc = ysect.plot_array(head, head=head, alpha=0.5, ax=ax)
#    pc1 = ysect.contour_array(head, head=head, alpha=0.5, ax=ax)#, levels=np.arange(1, 10, 1))
    patches = ysect.plot_ibound(head=head, ax=ax)
    linecollection = ysect.plot_grid(ax=ax)
    cb = plt.colorbar(pc, shrink=0.75, ax=ax)

    patches = ysect.plot_bc("WEL", kper=plot_kper[iplot], color="pink")
    patches = ysect.plot_ibound()
    linecollection = ysect.plot_grid()

plt.show()


fig = plt.figure(figsize=(10, 6))


plt.plot(times, subs_t[:,50], "-o", label="Total Subsidence")
plt.plot(times, comp1_t[:,50], "-o", label="Layer 1 compaction")
plt.plot(times, comp2_t[:,50], "-o", label="Layer 2 compaction")
plt.plot(times, comp3_t[:,50], "-o", label="layer 3 compaction")
plt.xlabel("time (days)")
plt.ylabel("subsidence(m)")
plt.legend()

plt.show()

# 畫出各層的水頭值
extent = (delr / 2.0, Lx - delr / 2.0, Ly - delc / 2.0, delc / 2.0)

fig = plt.figure(figsize=(15, 6))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])
ax4 = plt.subplot(gs[1, :])

comp = sobj1.get_data(totim=times[1])
print(comp[0, :, :])
ax1.contourf(comp[0, :, :], levels=20, extent=extent)
ax1.set_title("layer 1 compaction")
ax2.contourf(comp[1, :, :], levels=20, extent=extent)
ax2.set_title("layer 2 compaction")
ax3.contourf(comp[2, :, :], levels=20, extent=extent)
ax3.set_title("layer 3 compaction")
ax4.plot(times, subs_t[:,50], "-o", label="Total Subsidence")
ax4.plot(times, comp1_t[:,50], "-o", label="Layer 1 compaction")
ax4.plot(times, comp2_t[:,50], "-o", label="Layer 2 compaction")
ax4.plot(times, comp3_t[:,50], "-o", label="Layer 3 compaction")
ax4.set_xlabel("time (days)")
ax4.set_ylabel("subsidence(m)")
ax4.legend()

# 調整圖之間的間距
#plt.tight_layout()


plt.show()


try:
    temp_dir.cleanup()
except:
    # prevent windows permission error
    pass