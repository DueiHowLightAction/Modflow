##### Plot the data from workspace #####

from pathlib import Path
import json
from tempfile import TemporaryDirectory
import numpy as np
import flopy
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf
import os
import sys
import matplotlib.gridspec as gridspec
import pandas as pd

ncol = 200


workspace1 = Path(r"D:\NTU_Stride-C\codes\modflow\workspace\SUB-Yulin-5_0")

# 讀取各參數的模擬結果
# 對第一層參數的敏感度測試
sobj0 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\sfe0\SUB-Yulin-5_0.total_comp.hds",
                   text="COMPACTION")
sobj1 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\sfe1\SUB-Yulin-5_0.total_comp.hds",
                   text="COMPACTION")
sobj2 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\sfe2\SUB-Yulin-5_0.total_comp.hds",
                   text="COMPACTION")
sobj00 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\sfe00\SUB-Yulin-5_0.total_comp.hds",
                   text="COMPACTION")
sobj01 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\sfe01\SUB-Yulin-5_0.total_comp.hds",
                   text="COMPACTION")
# 第二層參數的敏感度測試
sobj2_0 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\layer 2\sfe0\SUB-Yulin-5_0.total_comp.hds",
                   text="COMPACTION")
sobj_0 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\layer 2\sfe0\SUB-Yulin-5_0.subsidence.hds",
                   text="SUBSIDENCE")
sobj2_1 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\layer 2\sfe1\SUB-Yulin-5_0.total_comp.hds",
                   text="COMPACTION")
sobj_1 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\layer 2\sfe1\SUB-Yulin-5_0.subsidence.hds",
                   text="SUBSIDENCE")
sobj2_2 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\layer 2\sfe2\SUB-Yulin-5_0.total_comp.hds",
                   text="COMPACTION")
sobj_2 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\layer 2\sfe2\SUB-Yulin-5_0.subsidence.hds",
                   text="SUBSIDENCE")
sobj2_3 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\layer 2\sfe3_nn=80\SUB-Yulin-5_0.total_comp.hds",
                   text="COMPACTION")
sobj_3 = bf.HeadFile(r"D:\NTU_Stride-C\codes\modflow\sensitivity_study\layer 2\sfe3_nn=80\SUB-Yulin-5_0.subsidence.hds",
                   text="SUBSIDENCE")
times = sobj0.get_times()

# 300 m井測資料
HLES = [0, 1.1, 1.9, 2.7, 3.3, 4.2, 4.7]#, 4.9, 0, 0, 0]
STES = [0, 1.3, 2.2, 3.7, 4.1, 5.3, 6.1]#, 6.3, 0, 0, 0]
TKES = [0, 1.5, 2.3, 3.2, 3.6, 5.4, 6.1]#, 5.8, 0, 0, 0]
for i in range(len(HLES)):
    HLES[i] = -HLES[i]*5/1.5/100
    STES[i] = -STES[i]*5/1.15/100
    TKES[i] = -TKES[i]*5/1.5/100

# GPS資料
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


sfe0 = 1.3e-4       # 宏崙井最小的
sfe00 = 4.6e-4
sfe1 = 1.14286e-3   # (0.00081+0.00059+0.00192+0.00022+0.00046+0.00013+0.00387)/7
sfe2 = 2.53e-3 
sfe3 = 5.0e-3
 
print("times steps are: ", times)
#subs_t = np.zeros((len(times), ncol))
comp1_t_0 = np.zeros((len(times), ncol))
comp1_t_00 = np.zeros((len(times), ncol))
comp1_t_01 = np.zeros((len(times), ncol))
comp1_t_1 = np.zeros((len(times), ncol))
comp1_t_2 = np.zeros((len(times), ncol))
comp2_t_0 = np.zeros((len(times), ncol))
comp2_t_1 = np.zeros((len(times), ncol))
comp2_t_2 = np.zeros((len(times), ncol))
comp2_t_3 = np.zeros((len(times), ncol))
subs_0 = np.zeros((len(times), ncol))
subs_1 = np.zeros((len(times), ncol))
subs_2 = np.zeros((len(times), ncol))
subs_3 = np.zeros((len(times), ncol))
#comp4_t = np.zeros((len(times), ncol))

for i in range(len(times)):
    for j in range(ncol):
#        subs_t[i, j] = sobj.get_data(totim=times[i])[0,0,j]
        comp1_t_0[i, j] = sobj0.get_data(totim=times[i])[0,0,j]
        comp1_t_00[i, j] = sobj00.get_data(totim=times[i])[0,0,j]
        comp1_t_01[i, j] = sobj01.get_data(totim=times[i])[0,0,j]
        comp1_t_1[i, j] = sobj1.get_data(totim=times[i])[0,0,j]
        comp1_t_2[i, j] = sobj2.get_data(totim=times[i])[0,0,j]
        comp2_t_0[i, j] = sobj2_0.get_data(totim=times[i])[1,0,j]
        comp2_t_1[i, j] = sobj2_1.get_data(totim=times[i])[1,0,j]
        comp2_t_2[i, j] = sobj2_2.get_data(totim=times[i])[1,0,j]
        comp2_t_3[i, j] = sobj2_3.get_data(totim=times[i])[1,0,j]
        subs_0[i, j] = sobj_0.get_data(totim=times[i])[0,0,j]
        subs_1[i, j] = sobj_1.get_data(totim=times[i])[0,0,j]
        subs_2[i, j] = sobj_2.get_data(totim=times[i])[0,0,j]
        subs_3[i, j] = sobj_3.get_data(totim=times[i])[0,0,j]
#        comp4_t[i, j] = sobj1.get_data(totim=times[i])[3,0,j]

fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.plot(times, -comp1_t_0[:,49], "-o", label="Ske=1.3e-4")
ax1.plot(times, -comp1_t_00[:,49], "-o", label="ske=4.6e-4")
ax1.plot(times, -comp1_t_01[:,49], "-o", label="ske=5.9e-4")
ax1.plot(times, -comp1_t_1[:,49], "-o", label="Ske=1.143e-3")
ax1.plot(times, -comp1_t_2[:,49], "-o", label="Ske=2.53e-3")
ax1.plot(times, HLES, "o", color="0", label="Well Observation")
ax1.set_title("Subsidence above 300 m (Layer 1)")
ax1.set_xlabel("time (days)")
ax1.set_ylabel("subsidence(m)")
ax1.set_ylim(ymin=min(obs_subs)/1000)

ax2.plot(times, -comp1_t_01[:,49], "-o", label="layer 1")
ax2.plot(times, -subs_0[:,49], "-o", label="Ske2=1.3e-4")
ax2.plot(times, -subs_1[:,49], "-o", label="Ske2=1.143e-3")
ax2.plot(times, -subs_2[:,49], "-o", label="Ske2=2.53e-3")
ax2.plot(times, -subs_3[:,49], "-o", label="Ske2=5.0e-3")
ax2.plot(times, obs_subs/1000, "o", color="0", label="GPS Data")
ax2.set_title("Total Subsidence")
ax2.set_xlabel("time (days)")
ax2.set_ylabel("subsidence(m)")
ax2.set_ylim(ymin=min(obs_subs)/1000)

ax2.legend()
ax1.legend()
plt.tight_layout()
plt.show()